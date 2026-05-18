class AppState:

    def __init__(self):

        self.video_path = None
        self.mat_path = None

        self.cap = None

        self.fps = 30
        self.total_frames = 0
        self.duration = 0

        self.current_time = 0.0

        self.start_time = 0.0
        self.end_time = 10.0

        self.playback_speed = 1.0

        self.is_playing = False

        self.t = None
        self.plot_signals = []

        self.data_type = "IMU"