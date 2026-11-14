"""Text runs of a PDF page with their boxes (PDF points, origin top-left), via pdfium."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Seg:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def h(self) -> float:
        return self.bbox[3] - self.bbox[1]


def page_segments(pdf_path: Path, page: int = 0) -> tuple[tuple[float, float], list[Seg]]:
    """(page width, height) and one Seg per text run that pdfium reports on the page."""
    import pypdfium2 as pdfium

    from app.concurrency import pdfium_lock

    with pdfium_lock:  # pdfium is not thread-safe, not even across documents
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            pg = pdf[page]
            w, h = pg.get_size()
            tp = pg.get_textpage()
            segs = []
            for i in range(tp.count_rects(0, -1)):
                left, bottom, right, top = tp.get_rect(i)
                t = " ".join(tp.get_text_bounded(left, bottom, right, top).split())
                if t:
                    segs.append(Seg(t, (left, h - top, right, h - bottom)))
            return (w, h), segs
        finally:
            pdf.close()
