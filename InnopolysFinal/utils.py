from PIL import Image
import base64
import io

def img2byte(im):
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data