"""Example plugin: reports when a rule's objects appear or disappear, and shows the count."""
import time

import cv2

from colorvision import Plugin


class TargetWatch(Plugin):
    name = "target_watch"
    rule = "hp_bar_green"

    def on_start(self, profile):
        self.visible = None

    def on_frame(self, result):
        visible = bool(result.detections.get(self.rule))
        if visible != self.visible:
            state = "widoczny" if visible else "brak"
            print(f"[{time.strftime('%H:%M:%S')}] {self.rule}: {state}")
            self.visible = visible

    def draw(self, canvas, result):
        count = len(result.detections.get(self.rule, []))
        cv2.putText(canvas, f"{self.rule}: {count}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
