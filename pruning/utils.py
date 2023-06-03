import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import time


class DummyProfiler:
    def __init__(self, label):
        self._label = label
        self._start = None
        self._sample = []

    def start(self):
        self._start = time.time()

    def stop(self):
        self._sample.append(time.time() - self._start)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def summary(self):
        sample = np.asarray(self._sample) * 1000.
        summary_str = f'Profiler <{self._label}> summary (ms):\n'
        summary_str += f'Samples:\t{len(sample)}\n'
        if len(sample):
            summary_str += f'max:\t{np.max(sample)}\n'
            summary_str += f'min:\t{np.min(sample)}\n'
            summary_str += f'std:\t{np.std(sample)}\n'
            summary_str += f'q95:\t{np.quantile(sample, 0.95)}\n'
            summary_str += f'q90:\t{np.quantile(sample, 0.9)}\n'
            summary_str += f'q50:\t{np.quantile(sample, 0.5)}\n'
            summary_str += f'avg:\t{np.mean(sample)}\n'
        return summary_str

    def __del__(self):
        print(self.summary())


def save_prediction_report(images, descriptions, output_file, img_size=None, summary=None, colors=None):
    table_html = '<table style="border-collapse: collapse;">'
    if summary:
        summary_row_html = f'<tr><td colspan="2" style="border: 1px solid black; text-align: center;">{summary}</td></tr>'
        table_html += summary_row_html
    
    for idx, (image, description) in enumerate(zip(images, descriptions)):
        image = Image.fromarray(image.astype(np.uint8))
        if img_size:
            image = image.resize(img_size)
        height, width = image.size
        image_file = io.BytesIO()
        image.save(image_file, format='PNG')
        image_file.seek(0)
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_tag = f'<img src="data:image/png;base64,{image_base64}" alt="CIFAR10 Image" style="width: {width}px; height: {height}px;">'
        background_color = ''
        if colors:
            color = colors[idx]
            background_color = f'background-color: rgb({color[0]}, {color[1]}, {color[2]});'
        row_html = f'<tr style="{background_color}"><td style="border: 1px solid black;">{image_tag}</td><td style="border: 1px solid black;">{description}</td></tr>'
        table_html += row_html
    table_html += '</table>'
    
    with open(output_file, 'w') as f:
        f.write(table_html)


def create_graph_image(sample, image_size=(300, 600)):
    matplotlib.use('Agg')
    height, width = image_size
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax.set_xlim([0, width])
    ax.set_ylim([0, 1.1 * np.max(sample)])
    ax.plot(np.arange(len(sample)) * (width / len(sample)), sample)

    fig.canvas.draw()
    graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return graph_image


def plot_image_matrix(image_list, row_size):
    num_images = len(image_list)
    num_rows = (num_images + row_size - 1) // row_size

    fig, axes = plt.subplots(num_rows, row_size, figsize=(15, 5 * num_rows))

    for i, image in enumerate(image_list):
        row = i // row_size
        col = i % row_size
        ax = axes[row, col] if num_rows > 1 else axes[col]

        ax.imshow(image)
        ax.axis('off')

    for j in range(num_images, num_rows * row_size):
        row = j // row_size
        col = j % row_size
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.show()
