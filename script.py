import cv2
import numpy as np
import urllib.request
import requests

# def url_to_image(url):
#     resp = urllib.request.urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     return image

# Example usage
url = 'https://media.discordapp.net/ephemeral-attachments/1211813163142479953/1337032438450356254/captcha.png?ex=67a5f7fe&is=67a4a67e&hm=f6e677d10f444ab043baf53cbe4f696db28fd1535b82fce8cce93a79ce36ed8d&=&format=webp&quality=lossless'
# image = url_to_image(url)

resp = requests.get(url)
image = np.asarray(bytearray(resp.content), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
print(image.shape)