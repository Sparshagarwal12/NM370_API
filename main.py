from fastapi import FastAPI
from pysolar.solar import *
from pydantic import BaseModel
import pandas._libs.tslibs.np_datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image
import cv2
from statistics import mean
from matplotlib import style
import matplotlib.pyplot as plt
import math
import base64
import io
import histogram as hist
from scipy.optimize import leastsq

# Instance of FastAPI class
app = FastAPI()

model = load_model('../water/DL/model.h5')


class SunModel(BaseModel):
    latitude: float
    longitude: float


class SunTurbid(BaseModel):
    image: str
    lat: float
    longitude: float


class AirModel(BaseModel):
    zenith1: float
    zenith2: float
    zenith3: float
    zenith4: float
    zenith5: float
    image1: str
    image2: str
    image3: str
    image4: str
    image5: str


class WaterModel(BaseModel):
    image: str


class TurbidModel(BaseModel):
    skyImage: str
    waterImage: str
    greyImage: str
    DN_s: int
    DN_w: int
    DN_c: int
    alpha: float
    S: int

# Air


def get_intencity(image: str):
    # Path tto be changed
    # path="C:/Users/om/Desktop/sih\\test"+str(image_no)+".jpg"
    # print(path)
    image = stringToRGB(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return (image[maxLoc])
    # display the results of our newly improved method


def get_am(zenith):
    return (1/math.cos(zenith))


def get_od(xs, ys):
    m = (-1*(math.log(mean(ys))-math.log(255))/mean(xs))
    return m


def get_red_alpha(t, t_o,):
    l = 630
    l_o = 450
    a = math.log(t/t_o)/math.log(l/l_o)
    return a


def get_blue_alpha(t, t_o):
    l = 450
    l_o = 520
    a = math.log(t/t_o)/math.log(l/l_o)
    return a


def get_green_alpha(t, t_o):
    l = 520
    l_o = 450
    a = math.log(t/t_o)/math.log(l/l_o)
    return a


def get_log(a):
    return math.log(a)


def red_plot_graph(xs, ys, od):
    b = 255
    func = np.vectorize(get_log)
    ys = func(ys)
    regression_line = [(math.log(b)-(od*x)) for x in xs]

    style.use('ggplot')
    plt.scatter(xs, ys, color='#FF0000')
    plt.plot(xs, regression_line)
    plt.xlabel("Air mass")
    plt.ylabel("Log(Intensity)")
    plt.title("For red channel langley plot")
    plt.savefig('red.png')


def blue_plot_graph(xs, ys, od):
    b = 255
    func = np.vectorize(get_log)
    ys = func(ys)
    regression_line = [(math.log(b)-(od*x)) for x in xs]

    style.use('ggplot')
    plt.scatter(xs, ys, color='#0000FF')
    plt.plot(xs, regression_line)
    plt.xlabel("Air mass")
    plt.ylabel("Log(Intensity)")
    plt.title("For blue channel langley plot")
    plt.savefig('blue.png')


def green_plot_graph(xs, ys, od):
    b = 255
    func = np.vectorize(get_log)
    ys = func(ys)
    regression_line = [(math.log(b)-(od*x)) for x in xs]

    style.use('ggplot')
    plt.scatter(xs, ys, color='#32CD32')
    plt.plot(xs, regression_line)
    plt.xlabel("Air mass")
    plt.ylabel("Log(Intensity)")
    plt.title("For green channel langley plot")
    plt.savefig('green.png')

# Water


def crop_img(image):
    y, x, s = image.shape
    cx = x//2
    cy = y//2
    image_cropped = image[cy-100:cy+100, cx-100:cx+100]
    return image_cropped


def mean(r):
    return np.mean(r)


def radiance(DN, alpha=1/4, S=100):
    L = DN/(S*alpha)
    return L


def reflectance(Ls, Lw, Lc):
    p = 3.14159265/0.18
    Rrs = (Lw-(0.028*Ls))/(p*Lc)
    return Rrs


def turbidity(Rrs):
    turb = (22.57*Rrs)/(0.044 - Rrs)
    return turb


def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def RGBTostring(name: str):
    with open(name, "rb") as img_file:
        baseString = base64.b64encode(img_file.read())
    return baseString.decode('utf-8')


@app.post("/")
async def root():
    return {"message": "dummy"}


@app.post("/sun")
async def sun(apiModel: SunModel):
    date = datetime.datetime.now(tz=datetime.timezone.utc)
    azimuth = get_azimuth(apiModel.latitude, apiModel.longitude, date)
    altitude = get_altitude(apiModel.latitude, apiModel.longitude, date)
    return {"azimuth": azimuth, "altitude": altitude, "zenith": 90 - altitude}


@app.post("/sunTurbid")
async def sunTurb(sun: SunTurbid):
    date = datetime.datetime.now(tz=datetime.timezone.utc)
    img = stringToRGB(sun.image)
    img = cv2.resize(img, (160, 120))

    a = -1
    b = -0.32
    c = 10
    d = -3
    e = 0.45

    def Zenith_Sky(up, vp):
        return math.acos((vp*math.sin(Z_camera)+fc*math.cos(Z_camera))/(math.sqrt(fc**2+up**2+vp**2)))

    def Azimuth_Sky(up, vp):
        return math.atan((fc*math.sin(A_camera)*math.sin(Z_camera)-up*math.cos(A_camera)-vp*math.sin(A_camera)*math.cos(Z_camera))/(fc*math.cos(A_camera)*math.sin(Z_camera)+up*math.sin(A_camera)-vp*math.cos(A_camera)*math.cos(Z_camera)))

    def Angle_sun_sky(Z_sun, Z_sky, A_sky, A_sun):
        return math.acos((math.cos(Z_sun)*math.cos(Z_sky))+(math.sin(Z_sun)*math.sin(Z_sky)*math.cos(A_sky-A_sun)))

    k = 0.5

    A_sun = math.radians(get_azimuth(sun.lat, sun.longitude, date))
    Z_sun = math.radians(90 - get_altitude(sun.lat, sun.longitude, date))

    A_camera = 0
    Z_camera = 1.57
    fc = 25
    Ip = []
    L = []
    Scaled = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            B, G, R = img[i, j]
            I = 0.2126*R+0.7152*G+0.0722*B
    #         print(I)
            Ip.append(I)
            Z_sky = Zenith_Sky(i, j)
            A_sky = Azimuth_Sky(i, j)
            Angle = Angle_sun_sky(Z_sun, Z_sky, A_sky, A_sun)

            g = (1+a*math.exp(b/math.cos(Z_sky))) * \
                (1+c*math.exp((d*math.cos(Angle))+e*(math.cos(Angle)**2)))
    #         print(g)
            Scaled.append(k*g)
            L.append((I-k*g)**2)

    Sca = np.array(Scaled)
    IP = np.array(Ip)

    def my_func(C, x, y):
        return IP - Sca

    starting_guess = np.ones((2, 1))
    data = (IP, Sca)

    result = leastsq(my_func, starting_guess, args=data)
    print(result)

    solution = result[0]
    print(solution[0])
    return {"solution": solution[0], "result": result}


@app.post("/air")
async def air(airModel: AirModel):

    x = []
    # Zenith angle fetech from app
    zenith = [airModel.zenith1, airModel.zenith2,
              airModel.zenith3, airModel.zenith4, airModel.zenith5]
    for i in zenith:
        x.append(get_am(i))
    y = []
    # do not change
    image = [airModel.image1, airModel.image2,
             airModel.image3, airModel.image4, airModel.image5]
    for i in image:
        a = get_intencity(i)
        y.append(a)

    x = np.array(x)
    y = np.array(y)
    print(x)

    red = y[:, 2]
    green = y[:, 1]
    blue = y[:, 0]
    print(red)
    print(y)
    od_red = get_od(x, red)
    od_green = get_od(x, green)
    od_blue = get_od(x, blue)
    red_alpha = get_red_alpha(od_red, od_blue)
    green_alpha = get_green_alpha(od_green, od_blue)
    blue_alpha = get_blue_alpha(od_blue, od_green)
    red_plot_graph(x, red, od_red)
    green_plot_graph(x, red, od_green)
    blue_plot_graph(x, red, od_blue)

    red = RGBTostring("red.png")
    green = RGBTostring("green.png")
    blue = RGBTostring("blue.png")

    return {"RedChannelAlpha": red_alpha, "GreenChannelAlpha": green_alpha, "BlueChannelAlpha": blue_alpha, "OpticalDepthRed": od_red, "OpticalDepthBlue": od_blue, "OpticalDepthRed": od_red, "red": red, "green": green, "blue": blue}


@app.post("/water")
async def water(apiModel: WaterModel):
    # if(not apiModel.imageurl):
    #     return {"message": "No Image url passed"}
    # os.system("wget -O image.jpg " + apiModel.imageurl)
    # imgpath = 'a.jpg'
    image = stringToRGB(apiModel.image)
    orig = image.copy()
    (h, w) = image.shape[:2]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    (h, l, m) = model.predict(img)[0]

    val = max(h, l, m)
    if val == h:
        label = "HIGH"
    elif val == l:
        label = "LOW"
    else:
        label = "MEDIUM"
    color = (0, 0, 255)

    label = "{}: {:.2f}%".format(label, max(h, l, m) * 100)
    return {"message": label}


@app.post("/turbidity")
async def turbid(turbidModel: TurbidModel):
    img_s = stringToRGB(turbidModel.skyImage)
    hist.hist(img_s, "sky.png")
    img_s = crop_img(img_s)
    b_s, g_s, r_s = cv2.split(img_s)

    img_w = stringToRGB(turbidModel.waterImage)
    hist.hist(img_w, "water.png")
    img_w = crop_img(img_w)
    b_w, g_w, r_w = cv2.split(img_w)

    img_c = stringToRGB(turbidModel.greyImage)
    hist.hist(img_c, "grey.png")
    img_c = crop_img(img_c)
    b_c, g_c, r_c = cv2.split(img_c)

    if turbidModel.DN_s is None:
        Rs = mean(r_s)
    else:
        Rs = turbidModel.DN_s
    if turbidModel.DN_w is None:
        Rw = mean(r_w)
    else:
        Rw = turbidModel.DN_w
    if turbidModel.DN_c is None:
        Rc = mean(r_c)
    else:
        Rc = turbidModel.DN_c

    if (turbidModel.alpha, turbidModel.S) is (None, None):
        Ls = radiance(Rs)
        Lw = radiance(Rw)
        Lc = radiance(Rc)
    else:
        Ls = radiance(Rs, turbidModel.alpha, turbidModel.S)
        Lw = radiance(Rw, turbidModel.alpha, turbidModel.S)
        Lc = radiance(Rc, turbidModel.alpha, turbidModel.S)

    Rrs = reflectance(Ls, Lw, Lc)

    turbid = turbidity(Rrs)

    return {"turbidity": turbid, "waterHist": RGBTostring("water.png"), "skyHist": RGBTostring("sky.png"), "greyHist": RGBTostring("grey.png")}
