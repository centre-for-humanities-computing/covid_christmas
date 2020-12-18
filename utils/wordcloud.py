import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude

from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator


def tokenlist_wordcloud(tokenlist, mask_img, save_as=""):
    text = " ".join(tokenlist)
    # load image. This has been modified in gimp to be brighter and have more saturation.
    parrot_color = np.array(Image.open(mask_img))

    # subsample by factor of 3. Very lossy but for a wordcloud we don't really care.
    color = parrot_color[::1, ::1]

    # create mask  white is "masked out"
    mask = color.copy()
    mask[mask.sum(axis=2) == 0] = 255

    # some finesse: we enforce boundaries between colors so they get less washed out.
    # For that we do some edge detection in the image
    edges = np.mean([gaussian_gradient_magnitude(
        color[:, :, i] / 255., 2) for i in range(3)], axis=0)
    mask[edges > .08] = 255

    # create wordcloud. A bit sluggish, you can subsample more strongly for quicker rendering
    # relative_scaling=0 means the frequencies in the data are reflected less
    # acurately but it makes a better picture
    wc = WordCloud(max_words=3000, mask=mask, background_color="white",
                   max_font_size=40, random_state=42, relative_scaling=0)

    # generate word cloud
    wc.generate(text)

    # create coloring from image
    image_colors = ImageColorGenerator(color)
    wc.recolor(color_func=image_colors)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    if save_as:
        wc.to_file(save_as)
