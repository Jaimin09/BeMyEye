# import the necessary packages
from pytesseract import Output
import pytesseract
import cv2
from gtts import gTTS 
import os
# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to localize each area of text in the input image
path=r'C:\Users\DELL\Desktop\UmmulKiram\Tensorflow\image.jpg'
image = cv2.imread(path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
t=""
# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(results["conf"][i])
# filter out weak confidence text localizations
    if conf > 50 and len(text)>0:
        # display the confidence and text to our terminal
        t=t+format(text)+" "
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")
        
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw a bounding box around the text along
        # with the text itself
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 0, 255), 3)
if len(t)!=0:
    print(t)
    speech = gTTS(text = t, lang = 'en', slow = False)
    speech.save('text.mp3')
    os.system('start text.mp3')
    os.remove('text.mp3')
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

