import json, os, sys, pathlib, skimage, PIL.Image, itertools, event_warping, concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from scipy import ndimage

np.seterr(divide="ignore", invalid="ignore")
variance_loss = []
fun_idx = 0
SAVE = 1
WEIGHT = 1
VIDEOS = ["FN034HRTEgyptB_NADIR", "20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5"]
OBJECTIVE = ["variance", "weighted_variance"]
FILENAME = VIDEOS[1]
HEURISTIC = OBJECTIVE[1]
VELOCITY_RANGE = (-30, 30)
RESOLUTION = 30
TMAX = 20e6
RATIO = 0.00001
readfrom = "data/"
savefileto = "img/"
scalar_velocities = np.linspace(VELOCITY_RANGE[0], VELOCITY_RANGE[1], RESOLUTION)
nVel = len(scalar_velocities)


def deleteimg():
    # Check if the folder exists
    if os.path.exists(savefileto):
        # Get a list of all files in the folder
        files = os.listdir(savefileto)
        if len(files) > 0:
            # Iterate through the files and delete them
            for file in files:
                file_path = os.path.join(savefileto, file)
                os.remove(file_path)
            print("All files in the folder have been deleted.")
        else:
            print("No images found in this folder.")
    else:
        print("The specified folder does not exist.")


def without_most_active_pixels(events: np.ndarray, ratio: float):
    assert ratio >= 0.0 and ratio <= 1.0
    count = np.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    np.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    return events[
        count[events["x"], events["y"]] <= np.percentile(count, 100.0 * (1.0 - ratio))
    ]


width, height, events = event_warping.read_es_file(readfrom + FILENAME + ".es")
width += 1
height += 1
events = without_most_active_pixels(events, ratio=0.000001)
ii = np.where(np.logical_and(events["t"] > 1, events["t"] < (TMAX)))
events = events[ii]
t = (events["t"][-1] - events["t"][0]) / 1e6
edgepx = t


def mirror(image):
    mirrored_image = []
    height, width = len(image), len(image[0])
    for i in range(height):
        mirrored_row = []
        for j in range(width - 1, -1, -1):
            mirrored_row.append(image[i][j])
        mirrored_image.append(mirrored_row)
    return np.array(mirrored_image)


def correction(i, j, vx, vy, width, height):
    return {
        "1": (1, vx / width, vy / height),
        "2": vx / x[i, j],
        "3": vy / y[i, j],
        "4": vx / (-x[i, j] + width + vx),
        "5": vy / (-y[i, j] + height + vy),
        "6": (vx * vy) / (vx * y[i, j] + vy * width - vy * x[i, j]),
        "7": (vx * vy) / (vx * height - vx * y[i, j] + vy * x[i, j]),
    }


def alpha_1(warped_image):
    """
    Input:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx < w and vy < h. The conditions are designed based on the pixel's
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions = [
        (x > vx) & (x < width) & (y >= vy) & (y <= height),
        (x > 0) & (x < vx) & (y <= height) & (y >= ((vy * x) / vx)),
        (x >= 0) & (x <= width) & (y > 0) & (y < vy) & (y < ((vy * x) / vx)),
        (x >= width) & (x <= width + vx) & (y >= vy) & (y <= (((vy * (x - width)) / vx) + height)),
        (x > vx) & (x < width + vx) & (y > height) & (y > (((vy * (x - width)) / vx) + height)) & (y < height + vy),
        (x > width) & (x < width + vx) & (y >= ((vy * (x - width)) / vx)) & (y < vy),
        (x > 0) & (x < vx) & (y > height) & (y <= (((vy * x) / vx) + height)),
    ]

    for idx, condition in enumerate(conditions, start=1):
        i, j = np.where(condition)
        correction_func = correction(i, j, vx, vy, width, height)
        if idx == 1:
            warped_image[i + 1, j + 1] *= correction_func[str(idx)][0]
        else:
            warped_image[i + 1, j + 1] *= correction_func[str(idx)]

    warped_image[x > width + vx - edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height + vy - edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy * (x - width)) / vx) + edgepx] = 0
    warped_image[y > (((vy * x) / vx) + height) - edgepx] = 0
    warped_image[np.isnan(warped_image)] = 0
    return warped_image


def alpha_2(warped_image, section: int):
    """
    Input:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx > w and vy > h. The conditions are designed based on the pixel's
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions_1 = [
        (x >= width) & (x <= vx) & (y >= (vy * x) / vx) & (y <= (vy / vx) * (x - width - vx) + vy + height),
        (x > 0) & (x < width) & (y >= (vy * x) / vx) & (y < height),
        (x > 0) & (x <= width) & (y > 0) & (y < (vy * x) / vx),
        (x > vx) & (x < vx + width) & (y > vy) & (y <= (vy / vx) * (x - width - vx) + vy + height),
        (x > vx) & (x < vx + width) & (y > (vy / vx) * (x - width - vx) + vy + height) & (y < height + vy),
        (x > width) & (x <= vx + width) & (y >= (vy * (x - width)) / vx) & (y < vy) & (y < (vy * x) / vx) & (y > 0),
        (x > 0) & (x <= vx) & (y < (vy / vx) * x + height) & (y >= height) & (y > (vy / vx) * (x - width - vx) + vy + height),
    ]

    conditions_2 = [
        (x >= 0) & (x < vx + width) & (y > ((vy * (x - width)) / vx) + height) & (y < vy) & (y > height)  & (y < (vy * x) / vx),
        (x >= 0) & (x <= vx) & (y > (vy * x) / vx) & (y < height),
        (x >= 0) & (x < width) & (y >= 0) & (y < (vy * x) / vx) & (y < height),
        (x > width) & (x < vx + width) & (y <= ((vy * (x - width)) / vx) + height) & (y > vy),
        (x >= vx) & (x < vx + width) & (y > ((vy * (x - width)) / vx) + height) & (y < vy + height) & (y > vy),
        (x >= width) & (x <= vx + width) & (y > (vy / vx) * (x - width)) & (y < ((vy * (x - width)) / vx) + height) & (y > 0) & (y < vy),
        (x >= 0) & (x <= vx) & (y <= (vy / vx) * x + height) & (y > (vy / vx) * x) & (y > height) & (y <= height + vy),
    ]

    conditions = [conditions_1, conditions_2]
    for idx, condition in enumerate(conditions[section - 1], start=1):
        i, j = np.where(condition)
        correction_func = correction(i, j, vx, vy, width, height)
        if idx == 1:
            warped_image[i + 1, j + 1] *= correction_func[str(idx)][section]
        else:
            warped_image[i + 1, j + 1] *= correction_func[str(idx)]

    warped_image[x > width + vx - edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height + vy - edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy * (x - width)) / vx) + edgepx] = 0
    warped_image[y > (((vy * x) / vx) + height) - edgepx] = 0
    warped_image[np.isnan(warped_image)] = 0
    return warped_image


def draw_lines_and_text(d, vx, vy, width, height):
    lines = [
        (0, 0, 0, height + vy),
        (0, 0, width + vx, 0),
        (0, height + vy, width + vx, height + vy),
        (0, height, width + vx, height),
        (0, vy, width + vx, vy),
        (vx, 0, vx, height + vy),
        (width, 0, width, height + vy),
        (vx + width, 0, vx + width, height + vy),
    ]

    texts = [
        ((2, 0), "0"),
        ((2, height + vy - 10), "h+vy*t"),
        ((2, height), "h"),
        ((2, vy), "vy*t"),
        ((vx - 25, 0), "vx*t"),
        ((width - 10, 0), "w"),
        ((vx + width - 40, 0), "w+vx*t"),
    ]

    for line in lines:
        d.line(line, fill=128, width=1)

    for position, text in texts:
        d.text(position, text, fill=(0, 255, 0))


def saveimg(contrastmap, var, fun_idx, t, fieldy, fieldx, imgpath, draw=False):
    colormap = plt.get_cmap("magma")  # type: ignore
    gamma = lambda image: image ** (1 / 1)
    scaled_pixels = gamma(
        (contrastmap - contrastmap.min()) / (contrastmap.max() - contrastmap.min())
    )
    image = PIL.Image.fromarray(
        (colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8)
    )

    if draw:
        d = ImageDraw.Draw(image)
        draw_lines_and_text(d, vx, vy, width, height)

    new_image = image.resize((500, 500))
    filename = f"eventmap_f{fun_idx}_vy_{fieldy*t}_vx_{fieldx*t}_var_{var}.jpg"
    filepath = os.path.join(imgpath, filename)
    os.makedirs(imgpath, exist_ok=True)
    new_image.save(filepath)
    fun_idx = 0


def variance(evmap):
    flatimg = evmap.flatten()
    res = flatimg[flatimg != 0]
    return np.var(res)


if SAVE:
    deleteimg()

for iVely in range(0, nVel):
    fieldy = scalar_velocities[iVely]
    for iVelx in range(0, nVel):
        fun_id = 0
        fieldx = scalar_velocities[iVelx]
        velocity = (fieldx * 1e-6, fieldy * 1e-6)
        warped_image = event_warping.accumulate((width, height), events, velocity)
        vx = np.abs(fieldx * t)
        vy = np.abs(fieldy * t)
        corrected_warped_image = None

        x = np.tile(
            np.arange(1, warped_image.pixels.shape[1] + 1),
            (warped_image.pixels.shape[0], 1),
        )
        y = np.tile(
            np.arange(1, warped_image.pixels.shape[0] + 1),
            (warped_image.pixels.shape[1], 1),
        ).T

        if (
            fieldx * t >= 0.0
            and fieldy * t >= 0.0
            and np.abs(fieldx * t) <= width
            and np.abs(fieldy * t) <= height
        ) or (
            fieldx * t <= 0.0
            and fieldy * t <= 0.0
            and np.abs(fieldx * t) <= width
            and np.abs(fieldy * t) <= height
        ):
            fun_idx += 1
            corrected_warped_image = alpha_1(warped_image.pixels)

        if (
            fieldx * t >= 0.0
            and fieldy * t <= 0.0
            and np.abs(fieldx * t) <= width
            and np.abs(fieldy * t) <= height
        ) or (
            fieldx * t <= 0.0
            and fieldy * t >= 0.0
            and np.abs(fieldx * t) <= width
            and np.abs(fieldy * t) <= height
        ):
            fun_idx += 2
            warped_image.pixels = mirror(warped_image.pixels)
            corrected_warped_image = alpha_1(warped_image.pixels)

        if (
            (
                fieldx * t >= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) <= height
            )
            or (
                fieldx * t <= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) <= height
            )
            or (
                (((vy / vx) * width) - height) / (np.sqrt(1 + (vy / vx) ** 2)) <= 0
                and fieldx * t >= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
            or (
                (((vy / vx) * width) - height) / (np.sqrt(1 + (vy / vx) ** 2)) <= 0
                and fieldx * t <= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
        ):
            fun_idx += 3
            corrected_warped_image = alpha_2(warped_image.pixels, 1)

        if (
            (
                fieldx * t >= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) <= width
                and np.abs(fieldy * t) >= height
            )
            or (
                fieldx * t <= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) <= width
                and np.abs(fieldy * t) >= height
            )
            or (
                (((vy / vx) * width) - height) / (np.sqrt(1 + (vy / vx) ** 2)) > 0
                and fieldx * t >= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
            or (
                (((vy / vx) * width) - height) / (np.sqrt(1 + (vy / vx) ** 2)) > 0
                and fieldx * t <= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
        ):
            fun_idx += 4
            corrected_warped_image = alpha_2(warped_image.pixels, 2)

        if (
            (
                fieldx * t >= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) <= height
            )
            or (
                fieldx * t <= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) <= height
            )
            or (
                (height + vy - (vy / vx) * (width + vx))
                / (np.sqrt(1 + (-vy / vx) ** 2))
                >= 0
                and fieldx * t >= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
            or (
                (height + vy - (vy / vx) * (width + vx))
                / (np.sqrt(1 + (-vy / vx) ** 2))
                >= 0
                and fieldx * t <= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
        ):
            fun_idx += 5
            warped_image.pixels = mirror(warped_image.pixels)
            corrected_warped_image = alpha_2(warped_image.pixels, 1)

        if (
            (
                fieldx * t >= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) <= width
                and np.abs(fieldy * t) >= height
            )
            or (
                fieldx * t <= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) <= width
                and np.abs(fieldy * t) >= height
            )
            or (
                (height + vy - (vy / vx) * (width + vx))
                / (np.sqrt(1 + (-vy / vx) ** 2))
                < 0
                and fieldx * t >= 0.0
                and fieldy * t <= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
            or (
                (height + vy - (vy / vx) * (width + vx))
                / (np.sqrt(1 + (-vy / vx) ** 2))
                < 0
                and fieldx * t <= 0.0
                and fieldy * t >= 0.0
                and np.abs(fieldx * t) >= width
                and np.abs(fieldy * t) >= height
            )
        ):
            fun_idx += 6
            warped_image.pixels = mirror(warped_image.pixels)
            corrected_warped_image = alpha_2(warped_image.pixels, 2)

        var = variance(corrected_warped_image)
        # saveimg(corrected_warped_image, var, fun_idx, t, fieldy, fieldx, imgpath, draw=True)
        variance_loss.append(var)
        fun_idx = 0
        sys.stdout.write(
            f"\r{len(variance_loss)} / {RESOLUTION ** 2} ({(len(variance_loss) / (RESOLUTION ** 2) * 100):.2f} %)"
        )
        sys.stdout.flush()
    fp1 = 0
sys.stdout.write("\n")
with open(
    f"{savefileto + FILENAME}_{HEURISTIC}_{VELOCITY_RANGE[0]}_{VELOCITY_RANGE[1]}_{RESOLUTION}_{RATIO}_{WEIGHT}.json",
    "w",
) as output:
    json.dump(variance_loss, output)

variance_loss = np.reshape(variance_loss, (RESOLUTION, RESOLUTION))
variance_loss[np.isnan(variance_loss)] = 0
colormap = plt.get_cmap("magma")  # type: ignore
gamma = lambda image: image ** (1 / 2)
scaled_pixels = gamma(
    (variance_loss - variance_loss.min()) / (variance_loss.max() - variance_loss.min())
)
image = PIL.Image.fromarray((colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8))
im1 = image.rotate(90, PIL.Image.NEAREST, expand=1)  # type: ignore
new_im1 = im1.resize((400, 400))
im1 = new_im1.save(savefileto+ FILENAME+ "_"+ str(VELOCITY_RANGE[0])+ "_"+ str(VELOCITY_RANGE[1])+ "_"+ str(RESOLUTION)+ "_"+ str(RATIO)+ "_"+ str(WEIGHT)+ ".png")
new_im1.show()
