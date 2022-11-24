from PIL import Image


def generate_gradient(color):
    color = color.upper()
    w, h = 1000, 500
    r, g, b = 0, 0, 0
    counter = 4
    if color == 'GRAY':
        im = Image.new("L", (w, h))
        pixels = im.load()
        for i in range(w):
            if i % counter == 0 and i != 0:
                r += 1
            for j in range(h):
                pixels[i, j] = r

    else:
        im = Image.new("RGB", (w, h), (0, 0, 0))
        pixels = im.load()
        if color == 'R':
            for i in range(w):
                if i % counter == 0 and i != 0:
                    r += 1
                for j in range(h):
                    pixels[i, j] = r, g, b

        elif color == 'G':
            for i in range(w):
                if i % counter == 0 and i != 0:
                    g += 1
                for j in range(h):
                    pixels[i, j] = r, g, b

        elif color == 'B':
            for i in range(w):
                if i % counter == 0 and i != 0:
                    b += 1
                for j in range(h):
                    pixels[i, j] = r, g, b

    im.show()

if __name__ == "__main__":
    gradient('GRAY')