"""Review image utilities for comparing source images with canonical sketches."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class ReviewImageComposer:
    def __init__(self, panel_size: tuple[int, int] = (480, 360)):
        self.panel_size = panel_size

    def compose(self, original_path: str, sketch_path: str, output_path: str, title: str = "") -> str:
        original = self._read_or_blank(original_path)
        sketch = self._read_or_blank(sketch_path)
        original = self._fit_panel(original)
        sketch = self._fit_panel(sketch)
        header_h = 42
        width = self.panel_size[0] * 2
        canvas = np.full((self.panel_size[1] + header_h, width, 3), 255, dtype=np.uint8)
        canvas[header_h:, : self.panel_size[0]] = original
        canvas[header_h:, self.panel_size[0] :] = sketch
        cv2.putText(canvas, "original", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "canonical sketch", (self.panel_size[0] + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        if title:
            cv2.putText(canvas, title[:60], (width // 2 - 80, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    def compose_text_panel(self, original_path: str, lines: list[str], output_path: str, title: str = "") -> str:
        original = self._fit_panel(self._read_or_blank(original_path))
        header_h = 42
        width = self.panel_size[0] * 2
        canvas = np.full((self.panel_size[1] + header_h, width, 3), 255, dtype=np.uint8)
        canvas[header_h:, : self.panel_size[0]] = original
        cv2.putText(canvas, "original", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "extraction / no canonical build", (self.panel_size[0] + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        y = header_h + 28
        x = self.panel_size[0] + 12
        for raw_line in lines[:14]:
            for line in self._wrap(raw_line, 52):
                cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
                y += 20
                if y > canvas.shape[0] - 12:
                    break
            if y > canvas.shape[0] - 12:
                break
        if title:
            cv2.putText(canvas, title[:60], (width // 2 - 80, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    def compose_evidence_panel(
        self,
        original_path: str,
        overlay_path: str,
        lines: list[str],
        output_path: str,
        title: str = "",
    ) -> str:
        original = self._fit_panel(self._read_or_blank(original_path))
        overlay = self._fit_panel(self._read_or_blank(overlay_path))
        header_h = 42
        width = self.panel_size[0] * 3
        canvas = np.full((self.panel_size[1] + header_h, width, 3), 255, dtype=np.uint8)
        canvas[header_h:, : self.panel_size[0]] = original
        canvas[header_h:, self.panel_size[0] : self.panel_size[0] * 2] = overlay
        cv2.putText(canvas, "original", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "visual overlay", (self.panel_size[0] + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "diagnosis", (self.panel_size[0] * 2 + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 2, cv2.LINE_AA)
        y = header_h + 28
        x = self.panel_size[0] * 2 + 12
        for raw_line in lines[:14]:
            for line in self._wrap(raw_line, 52):
                cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
                y += 20
                if y > canvas.shape[0] - 12:
                    break
            if y > canvas.shape[0] - 12:
                break
        if title:
            cv2.putText(canvas, title[:60], (width // 2 - 80, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    def _read_or_blank(self, path: str) -> np.ndarray:
        if path and Path(path).exists():
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        return np.full((self.panel_size[1], self.panel_size[0], 3), 255, dtype=np.uint8)

    def _fit_panel(self, img: np.ndarray) -> np.ndarray:
        target_w, target_h = self.panel_size
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        panel = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        panel[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return panel

    def _wrap(self, text: str, width: int) -> list[str]:
        words = str(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word[:width]
        if current:
            lines.append(current)
        return lines or [""]
