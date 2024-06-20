import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from model import NNPy
import tkinter as tk

model = NNPy()
model.load_state_dict(torch.load("model_state.pth"))
model.eval()

transformations = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # magic numbers based on the mean and std of the dataset of mnist
    transforms.Normalize((0.1307,), (0.3081,))
])

""" image_path = "test.png"
image = Image.open(image_path)

# Add batch dimension (1,1,28,28) (batch_size, channels, height, width)
input_tensor = transformations(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)


pdf = torch.nn.functional.softmax(output[0], dim=0)

plt.bar(range(10), pdf.numpy())
plt.xlabel('Digit')
plt.ylabel('Probability')
plt.title('Digit Classification Probabilities')
plt.show() """


class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer Paint Tool")

        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.grid(row=0, column=0, padx=20, pady=20)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.fig, self.ax_prob = plt.subplots(figsize=(5, 3))
        self.bar = None
        self.setup_graph()

        # Add a separate figure for the preview
        self.preview_fig, self.ax_preview = plt.subplots(figsize=(2, 2))
        self.ax_preview.axis("off")
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=root)
        self.preview_canvas.get_tk_widget().grid(row=1, column=0, padx=20, pady=20)

        self.update_graph()

        self.last_x, self.last_y = None, None

        # Clear Button
        self.clear_button = tk.Button(
            root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=20)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill='black', width=30, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill='black', width=30)
        self.last_x, self.last_y = x, y
        self.update_graph()

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.update_graph(clear=True)

    def setup_graph(self):
        self.ax_prob.set_title('Digit Classification Probabilities')
        self.ax_prob.set_xlabel('Digit')
        self.ax_prob.set_ylabel('Probability')
        self.ax_prob.set_xticks(range(10))
        self.ax_prob.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
        self.bar = self.ax_prob.bar(range(10), [0]*10)
        self.fig.canvas.draw()

        self.graph = FigureCanvasTkAgg(self.fig, master=self.root)
        self.graph.get_tk_widget().grid(row=0, column=1, padx=20, pady=20, rowspan=3)

    def update_graph(self, clear=False):
        if clear:
            probabilities = [0] * 10
            self.ax_preview.clear()
            self.ax_preview.axis("off")
        else:
            # Invert the image colors
            inverted_image = ImageOps.invert(self.image)
            processed_image = transformations(inverted_image).unsqueeze(0)
            with torch.no_grad():
                output = model(processed_image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Update preview
            self.ax_preview.clear()
            self.ax_preview.set_title('Normalized Input')
            self.ax_preview.axis("off")
            img = processed_image.squeeze().numpy()
            self.ax_preview.imshow(img, cmap="gray")

        for i, bar in enumerate(self.bar):
            bar.set_height(probabilities[i].item() if not clear else 0)

        self.fig.canvas.draw()
        self.preview_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
