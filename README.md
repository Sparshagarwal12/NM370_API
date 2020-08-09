# SERVER & REST API
Server built using Python and FastAPI to handle API requests and return requested calculated values to user


## DEPLOYMENT

> uvicorn main:app --host 0.0.0.0 --port 80


## ENDPOINTS

### **{url}/water**

> POST body
```
{
    image: (base64 encoded image String)
}
```

> response body
```
{
    "message": "turbidity"
}
```
### **{url}/sun**

> POST body
```
{
    latitude: float
    longitude: float
}
```

> Rseponse body
```
{
    "azimuth": float,
    "zenith": float,
    "altitude": float
}
```

### **{url}/turbidity**

> POST body
```
{
    skyImage: String(base64)
    waterImage: String(base64)
    greyImage: String(base64)
    DN_s: int
    DN_w: int
    DN_c: int
    alpha: float
    S: int
}
```

> Rseponse body
```
{
    turbidity: float,
    waterHist: String(base64),
    skyHist: String(base64),
    greyHist: String(base64)
}
```

### **{url}/air**

> POST body
```
{
    zenith1: float
    zenith2: float
    zenith3: float
    zenith4: float
    zenith5: float
    image1: String(base64)
    image2: String(base64)
    image3: String(base64)
    image4: String(base64)
    image5: String(base64)
}
```

> Rseponse body
```
{
    RedChannelAlpha: float
    GreenChannelAlpha: float
    BlueChannelAlpha: float
    OpticalDepthRed: float
    OpticalDepthGreen: float
    OpticalDepthBlue: float
    red: String(base64)
    green: String(base64)
    blue: String(base64)
}
```