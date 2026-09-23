"""학습 관측은 유지하고 사용자 화면에만 조작 안내를 표시한다."""
import time
import numpy as np
import gymnasium as gym
import pygame
from frame_pacing import FramePacer
from metrics import OFF_ROAD_LIMIT
from ui_text import translate, normalize_language
from gymnasium.envs.box2d.car_racing import CarRacing, WINDOW_W, WINDOW_H, TRACK_WIDTH, FPS

PANEL_HEIGHT = 330
ROAD_HEIGHT = WINDOW_H - 80
FONT_SIZE = 20


def fit_window(desktop):
    # SDL이 제공한 논리 화면 크기를 사용하고 제목줄/작업표시줄 여유를 둔다.
    width, height = desktop
    return max(320, min(1000, width - 80)), max(240, min(900, height - 120))


class GuidedCarRacing(CarRacing):
    gui_status = 'ready'
    gui_collect = True
    gui_language = 'kor'
    gui_frames = 0
    gui_notice = ''
    gui_notice_until = 0.0
    gui_model = ''
    gui_score_mode = False
    _gui_rendering = False

    def tr(self, key, **values):
        return translate(key, self.gui_language, **values)

    def current_notice(self):
        if not self.gui_notice or time.monotonic() >= self.gui_notice_until:
            return ''
        return self.tr(self.gui_notice)

    def show_notice(self, key):
        self.gui_notice = key
        self.gui_notice_until = time.monotonic() + 10.0

    def reset(self, **kwargs):
        from track import TRACK_SEED
        if kwargs.get('seed') is None:
            kwargs['seed'] = TRACK_SEED
        self.metrics = None
        self.clock = FramePacer()
        self.gui_frames = 0
        self._road_frame_key = None
        self._road_layer_key = None
        self._start_line = None
        self.gui_status = 'ready' if self.gui_collect else 'driving'
        observation, info = super().reset(**kwargs)
        from metrics import DrivingMetrics
        self.metrics = DrivingMetrics(self.track, self.car.hull.position, TRACK_WIDTH)
        return observation, info

    def step(self, action):
        metrics = getattr(self, 'metrics', None)
        if metrics is not None and (metrics.finished or metrics.aborted):
            return self.state, 0., True, False, metrics.result()
        # Update metrics before the human render, while keeping model pixels unchanged.
        mode = self.render_mode
        self.render_mode = None
        try:
            observation, _, terminated, truncated, info = super().step(action)
        finally:
            self.render_mode = mode
        if action is None:
            # osy의 출발선 형상. 완주/점수 판정과 독립적인 GUI 전용 표시다.
            beta = self.track[0][1]
            center = np.asarray(self.track[0][2:4])
            lateral = np.array([np.cos(beta), np.sin(beta)]) * TRACK_WIDTH
            offset = np.array([-np.sin(beta), np.cos(beta)]) * .35
            left, right = center - lateral, center + lateral
            self._start_line = [left - offset, right - offset, right + offset, left + offset]
        if action is not None and metrics is not None:
            position = self.car.hull.position
            on_road = any(fixture.TestPoint(position) for tile in self.road for fixture in tile.fixtures)
            metrics.update(position, on_road, 1. / FPS)
            terminated = metrics.finished or metrics.aborted
            info = metrics.result()
            if terminated:
                self.gui_status = 'finished' if metrics.finished else 'aborted'
        if mode == 'human':
            self.render()
        return observation, 0., terminated, truncated, info

    def _create_image_array(self, screen, size):
        # osy와 동일하게 GUI Surface에는 RGB 변환/축소를 적용하지 않는다.
        if self._gui_rendering:
            return None
        return super()._create_image_array(screen, size)

    def _render_road(self, zoom, translation, angle):
        # 관측/GUI 사이에 같은 도로 레이어를 재사용한다.
        key = (self.t, zoom, tuple(translation), angle)
        if key != self._road_layer_key:
            super()._render_road(zoom, translation, angle)
            self._road_layer = self.surf.copy()
            self._road_layer_key = key
        else:
            self.surf.blit(self._road_layer, (0, 0))
        if self._gui_rendering and self._start_line is not None:
            self._draw_colored_polygon(self.surf, self._start_line, (245, 245, 245),
                                       zoom, translation, angle)

    def _render(self, mode):
        if mode != 'human':
            return super()._render(mode)
        # 원본 하단 계기판은 잘라내고 사용자 안내는 별도 영역에 그린다.
        # 학습용 state_pixels에는 이 레이아웃을 적용하지 않는다.
        frame_key = (self.t, id(self._start_line))
        if self._road_frame_key != frame_key:
            self._gui_rendering = True
            try:
                super()._render('rgb_array')
            finally:
                self._gui_rendering = False
            self._gui_road_surface = self.surf
            self._road_frame_key = frame_key
        if self.screen is None:
            pygame.display.init()
            desktop = pygame.display.get_desktop_sizes()[0]
            self.screen = pygame.display.set_mode(fit_window(desktop), pygame.RESIZABLE)
        if not hasattr(self, '_guide_font'):
            fonts = {'kor': 'malgungothic,applegothic,nanumgothic,notosanscjkkr',
                     'eng': 'arial,dejavusans', 'chn': 'microsoftyahei,pingfangsc,heitisc,hiraginosansgb,songtisc,notosanscjksc,simsun'}
            font_path = pygame.font.match_font(fonts[self.gui_language])
            if font_path is None and self.gui_language != 'eng':
                raise RuntimeError(self.tr('missing_font'))
            self._guide_font = pygame.font.Font(font_path, FONT_SIZE)
            self._status_font = pygame.font.Font(font_path, 22)
        window_size = self.screen.get_size()
        compact = window_size[1] < 900
        road_height = 460 if compact else ROAD_HEIGHT
        panel_height = 240 if compact else PANEL_HEIGHT
        canvas = pygame.Surface((WINDOW_W, road_height + panel_height))
        canvas.fill((17, 23, 33))
        road = self._gui_road_surface.subsurface((0, 0, WINDOW_W, ROAD_HEIGHT))
        road_width = round(WINDOW_W * road_height / ROAD_HEIGHT)
        if road_height != ROAD_HEIGHT:
            road = pygame.transform.smoothscale(road, (road_width, road_height))
        canvas.blit(road, ((WINDOW_W - road.get_width()) // 2, 0))
        def text(value, x, row, color=(222, 231, 245), status=False):
            font = self._status_font if status else self._guide_font
            canvas.blit(font.render(value, True, color), (x, road_height + row))

        state = self.tr(self.gui_status if self.gui_status in ('ready', 'recording', 'driving', 'finished', 'aborted') else 'ready')
        if self.gui_collect and self.gui_status == 'finished':
            state += self.tr('unsaved')
        notice = self.current_notice()
        if notice:
            if self._status_font.size(state + '  |  ' + notice)[0] > WINDOW_W - 60:
                state = state.split(' · ')[0].split(' - ')[0]
            state += '  |  ' + notice
        metrics = getattr(self, 'metrics', None)
        recovering = metrics is not None and metrics.off_road and not metrics.aborted
        if recovering:
            remaining = max(0., OFF_ROAD_LIMIT - metrics.off_road_seconds)
            state = self.tr('recovering', seconds=remaining)
        color = ((104, 231, 179) if self.gui_status == 'finished' else
                 (243, 191, 91) if recovering or self.gui_status in ('ready', 'aborted') else (235, 242, 251))
        recording = self.gui_collect and self.gui_status == 'recording'
        light = ((255, 65, 78) if pygame.time.get_ticks() % 1000 < 500 else (89, 38, 46)) if recording else (90, 104, 123)
        pygame.draw.circle(canvas, light, (24, road_height + 25), 7)
        text(state, 42, 10, color, status=True)

        # 상태 다음 두 줄은 비워 두고 고정 열 좌표로 점수 표를 정렬한다.
        columns = (18, 235, 420, 645)
        table_top = 48 if compact else 100
        headers = (self.tr('metric_header'), self.tr('total_header'),
                   self.tr('gain_header'), self.tr('rule_header'))
        for column, (x, header) in enumerate(zip(columns, headers)):
            if not compact or column < 3:
                text(header, x, table_top, (153, 174, 201))
        metrics = getattr(self, 'metrics', None)
        total = metrics.display_totals if metrics else (0., 0., 0.)
        delta = metrics.display_delta if metrics else (0., 0., 0.)
        lap = metrics.lap_time if metrics else None
        record = self.tr('lap_record', seconds=lap) if lap is not None else self.tr('lap_pending')
        rows = [
            (self.tr('progress'), f'{total[0]:.2f} / 100', f'+{delta[0]:.2f}',
             self.tr('progress_rule')),
            (self.tr('centering'), f'{total[1]:.2f} / 100', f'+{delta[1]:.2f}',
             self.tr('centering_rule')),
            (self.tr('lap_time'), f'{total[2]:.2f}s', '', record),
        ]
        for index, row in enumerate(rows):
            y = table_top + 34 + index * 34
            for column, (x, value) in enumerate(zip(columns, row)):
                if not compact or column < 3:
                    text(value, x, y)
        for row in (table_top + 28, table_top + 64, table_top + 98, table_top + 134):
            pygame.draw.line(canvas, (43, 55, 72), (18, road_height + row), (WINDOW_W - 18, road_height + row))

        if self.gui_collect:
            detail = (self.tr('start') if self.gui_status == 'ready' else '')
            detail += self.tr('drive_keys')
            commands = (self.tr('save_keys') if self.gui_status == 'finished' else
                        self.tr('save_later_keys'))
            commands += self.tr('exit_keys')
        else:
            detail = self.tr('model') + self.gui_model
            commands = (self.tr('benchmark_keys') if self.gui_score_mode else
                        self.tr('infer_keys'))
        text(detail, 18, panel_height - 66)
        text(commands, 18, panel_height - 36)
        pygame.display.set_caption(self.tr('collect_title' if self.gui_collect else 'infer_title'))
        pygame.event.pump()
        self.clock.tick(self.metadata['render_fps'])
        scale = min(window_size[0] / canvas.get_width(), window_size[1] / canvas.get_height())
        size = (max(1, int(canvas.get_width() * scale)), max(1, int(canvas.get_height() * scale)))
        self.screen.fill((17, 23, 33))
        self.screen.blit(pygame.transform.smoothscale(canvas, size),
                         ((window_size[0] - size[0]) // 2, (window_size[1] - size[1]) // 2))
        pygame.display.flip()


def make_driving_env(display=True, collect=False, unlimited=False, language='kor'):
    language = normalize_language(language)
    env_id = 'ImitationCarRacing-v0'
    if env_id not in gym.envs.registry:
        gym.register(env_id, entry_point=GuidedCarRacing, max_episode_steps=1000)
    env = gym.make(env_id, render_mode='human' if display else 'rgb_array',
                   max_episode_steps=-1 if collect or unlimited else 1000)
    env.unwrapped.gui_collect = collect
    env.unwrapped.gui_language = language
    return env
