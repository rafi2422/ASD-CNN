import tempfile

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import math
from PIL import Image
import glob

margin = 0.1
saveDest = 'D:/Autism-Data/Facebook/RawFaces/Autistic/'
count = len(glob.glob(saveDest+'*'))

def draw_faces(filename, result_list):
    global count
    for i in range(len(result_list)):
        img = Image.open(filename)
        Iwidth, Iheight = img.size
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height

        x1 = max(math.floor(x1 * (1-margin)),0)
        y1 = max(math.floor(y1 * (1-margin)),0)
        x2 = min(math.floor(x2 * (1+margin)), Iwidth)
        y2 = min(math.floor(y2 * (1+margin)), Iheight)

        im_crop = img.crop((x1, y1, x2, y2)).save(saveDest + str(count) + '.jpg', quality=100)
        count += 1


detector = MTCNN()

filestoMTCNN = glob.glob('D:/Autism-Data/Facebook/Raw/Autistic/*')
for file in filestoMTCNN:
    pixels = pyplot.imread(file)
    faces = detector.detect_faces(pixels)
    draw_faces(file, faces)
    pass
