import cv2
from PIL import Image, ImageTk


class VideoPanel:

    def __init__(self, parent, state):

        self.parent = parent
        self.state = state

        self.label = None

    def set_widget(self, label_widget):
        self.label = label_widget

    def load_video(self, path):

        self.state.video_path = path

        self.state.cap = cv2.VideoCapture(path)

        self.state.fps = self.state.cap.get(cv2.CAP_PROP_FPS)

        self.state.total_frames = int(
            self.state.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.state.duration = (
            self.state.total_frames / self.state.fps
        )

        self.state.end_time = self.state.duration

    def update_frame(self):

        if self.state.cap is None:
            return

        # clamp frame
        frame_number = int(self.state.current_time * self.state.fps)
        frame_number = max(0, min(frame_number, self.state.total_frames - 1))

        # only seek if necessary (IMPORTANT FIX)
        current_pos = int(self.state.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if abs(current_pos - frame_number) > 1:
            self.state.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.state.cap.read()

        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)

        # FIXED scaling prevents "zoom illusion"
        img = img.resize((600, 450), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)