import sys
import webbrowser
from urllib.parse import quote

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# 一些好用的解析接口
head_urls = [
    "https://bfq.txnp.cn/player?url=", # 支持腾讯视频
    "https://yparse.ik9.cc/index.php?url=", # 支持爱奇艺
]


# 按键响应的处理函数
def handle_button_click(url, txt):
    source_url = txt.text().strip()
    if not source_url:
        return
    # 避免原始 URL 中的 & 等字符打断解析接口参数
    full_url = url + quote(source_url, safe="")
    webbrowser.open(full_url, new=2)


# 按键响应
def button_click(line, txt):
    handle_button_click(head_urls[line - 1], txt)


# 输入框的文本重置清空
def ent_cls(ent):
    ent.clear()


class VideoParserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._is_topmost = False
        self.setWindowTitle("视频解析（仅供学习）")
        self.setFixedSize(780, 560)
        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        container = QVBoxLayout(root)
        container.setContentsMargins(20, 20, 20, 20)
        container.setSpacing(14)

        title = QLabel("视频解析工具")
        title.setObjectName("title")
        subtitle = QLabel("请输入视频网址并选择线路")
        subtitle.setObjectName("subtitle")

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        label = QLabel("视频网址")
        label.setObjectName("label")

        self.ent = QLineEdit()
        self.ent.setPlaceholderText("https://...")
        self.ent.setClearButtonEnabled(True)

        clear_btn = QPushButton("重置")
        clear_btn.clicked.connect(lambda: ent_cls(self.ent))

        self.top_btn = QPushButton("未置顶")
        self.top_btn.setObjectName("topBtn")
        self.top_btn.setProperty("topOn", False)
        self.top_btn.clicked.connect(self.toggle_topmost)

        top_row.addWidget(label)
        top_row.addWidget(self.ent, 1)
        top_row.addWidget(clear_btn)
        top_row.addWidget(self.top_btn)

        lines_card = QWidget()
        lines_card.setObjectName("card")
        grid = QGridLayout(lines_card)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        for i in range(len(head_urls)):
            btn = QPushButton(f"线路{i + 1}")
            btn.setObjectName("lineBtn")
            btn.clicked.connect(lambda _=False, line=i + 1: button_click(line, self.ent))
            row = i // 6
            col = i % 6
            grid.addWidget(btn, row, col)

        container.addWidget(title)
        container.addWidget(subtitle)
        container.addLayout(top_row)
        container.addWidget(lines_card, 1)

        self.statusBar().showMessage("就绪")

    def toggle_topmost(self):
        self._is_topmost = not self._is_topmost
        self.setWindowFlag(Qt.WindowStaysOnTopHint, self._is_topmost)
        self.show()

        self.top_btn.setProperty("topOn", self._is_topmost)
        self.top_btn.setText("已置顶" if self._is_topmost else "未置顶")
        self.top_btn.style().unpolish(self.top_btn)
        self.top_btn.style().polish(self.top_btn)
        self.top_btn.update()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f172a;
            }
            #title {
                color: #f8fafc;
                font-size: 26px;
                font-weight: 700;
                padding-top: 2px;
            }
            #subtitle {
                color: #94a3b8;
                font-size: 13px;
                padding-bottom: 6px;
            }
            #label {
                color: #cbd5e1;
                font-size: 13px;
                min-width: 56px;
            }
            QLineEdit {
                background-color: #111827;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 8px 10px;
                selection-background-color: #2563eb;
            }
            QLineEdit:focus {
                border: 1px solid #3b82f6;
            }
            #card {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 8px 10px;
            }
            QPushButton:hover {
                background-color: #273449;
                border: 1px solid #475569;
            }
            QPushButton:pressed {
                background-color: #334155;
            }
            #lineBtn {
                min-width: 88px;
            }
            #topBtn[topOn="false"] {
                background-color: #14532d;
                border: 1px solid #166534;
            }
            #topBtn[topOn="true"] {
                background-color: #7f1d1d;
                border: 1px solid #b91c1c;
            }
            QStatusBar {
                background-color: #0b1220;
                color: #94a3b8;
                border-top: 1px solid #1f2937;
            }
            """
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VideoParserWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
