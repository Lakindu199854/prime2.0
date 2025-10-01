import layoutparser as lp
from PIL import Image

# Use the correct model alias
model = lp.AutoLayoutModel("lp://EfficientDeteResNet50/PubLayNet")

if model is None:
    raise Exception("Model failed to load!")

print("Model loaded!")

# Load the image from local folder
image = Image.open("A6_page-0001.jpg")
layout = model.detect(image)

print(layout)


# Optional: draw the layout blocks on the image
image_with_blocks = lp.draw_box(image, layout, box_width=3)
image_with_blocks.show()  # opens the image in the default viewer

