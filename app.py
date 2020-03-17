import os,cv2,pytesseract
from flask import Flask, render_template, request,jsonify
from PIL import Image

import numpy as np
# import imutils
import time

# pytesseract.pytesseract.tesseract_cmd = '/usr/share/tesseract-ocr'

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

start_time = time.time()


def join_images(list_of_images):
    images = [Image.open(x) for x in list_of_images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    images = images[::-1]
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save('out.jpg')
    joined_text = pytesseract.image_to_string(new_im,
                                              config='--psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return (joined_text)


def read_each_image(image_name):
    # Load image, grayscale, Otsu's threshold
    image = cv2.imread(image_name)

    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # image = imutils.resize(image, width=500)

    # Remove border
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    temp1 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
    temp3 = cv2.add(temp1, temp2)

    result = cv2.add(temp3, image)
    # Convert to grayscale and Otsu's threshold
    gray = cv2.cvtColor(temp3, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and filter using contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # sort contours
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])
    cnts = cnts[::-1]
    list_of_images = []
    for i, c in enumerate(cnts):
        temp3 = cv2.medianBlur(temp3, 3)
        temp3[np.where(temp3 > [120])] = [255]

        x, y, w, h = cv2.boundingRect(c)
        try:
            cropped = temp3[y - 9:y + h + 9, x - 9: x + w + 9]
            s = UPLOAD_FOLDER + '/sub_images/_r_crop_' + str(i) + '.jpg'
            cv2.imwrite(s, cropped)
        except:
            cropped = temp3[y:y + h, x: x + w]
            s = UPLOAD_FOLDER + '/sub_images/_r_crop_' + str(i) + '.jpg'
            cv2.imwrite(s, cropped)

        #         cv2.rectangle(temp3, (x-9, y-9), (x + w+9, y + h+9), (255,0, 255), 2)

        list_of_images.append(s)

    each_word = join_images(list_of_images)
    return each_word


def captch_ex(file_name, is_gruopby):
    img = cv2.imread(file_name)

    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # cv.THRESH_BINARY_INV

    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    words_list = []
    index = 1
    if not os.path.exists(UPLOAD_FOLDER + '/sub_images'): os.makedirs(UPLOAD_FOLDER + '/sub_images')
    if is_gruopby:

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Get plot positives that are text
            if w < 35 and h < 35:
                continue
            # you can crop image and send to OCR  , false detected will return no text :)
            cropped = new_img[y:y + h, x: x + w]
            cv2.rectangle(new_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            s = UPLOAD_FOLDER + '/sub_images/new_img' + str(index) + '.jpg'
            cv2.imwrite(s, cropped)
            image_text = pytesseract.image_to_string(cropped)
            index = index + 1

            return image_text
    else:
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Get plot positives that are text
            if w < 55 and h > 35:
                # draw rectangle around contour on original image
                #             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

                # you can crop image and send to OCR  , false detected will return no text :)
                cropped = new_img[y:y + h, x: x + w]

                s = UPLOAD_FOLDER + '/sub_images/first_crop_' + str(index) + '.jpg'
                cv2.imwrite(s, cropped)
                word = read_each_image(s)
                words_list.append(word)
                index = index + 1

    return "---------------\n".join(words_list)


def read_text_from_image(file_name, small_text=False, vertical_text=False):
    if small_text:
        image_text = captch_ex(file_name, vertical_text)
    else:
        img = cv2.imread(file_name)
        # img = imutils.resize(img, width=500)

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # cv.THRESH_BINARY_INV

        image_text = pytesseract.image_to_string(new_img)
    return image_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ocr', methods=['POST','GET'])
def upload_file():
    if request.method == "GET":
        return "This is the api BLah blah"
    elif request.method == "POST":
        file = request.files['image']

        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
        file.save(f)
        # print(file.filename)

        small_text = False
        vertical_text = False
        # preprocess = request.form["preprocess"]
        # if  preprocess == "small_text": small_text = True
        # if preprocess == "vertical_text": vertical_text = True

        file_name = UPLOAD_FOLDER+"/"+file.filename
        text = read_text_from_image(file_name, small_text, vertical_text)
        if os.path.exists(UPLOAD_FOLDER + '/sub_images'): os.remove(UPLOAD_FOLDER + '/sub_images')
        os.remove(UPLOAD_FOLDER + "/" + file.filename)
        # image = cv2.imread(UPLOAD_FOLDER+"/"+file.filename)
        # os.remove(UPLOAD_FOLDER+"/"+file.filename)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we should apply thresholding to preprocess the
        # image
        # preprocess = request.form["preprocess"]
        # if  preprocess == "thresh":
        #     gray = cv2.threshold(gray, 0, 255,
        #                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #
        # # make a check to see if median blurring should be done to remove
        # # noise
        #
        # elif preprocess == "blur":
        #     gray = cv2.medianBlur(gray, 3)
        # print(preprocess)
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        # filename = "{}.png".format(os.getpid())
        # cv2.imwrite(filename, gray)
        # # load the image as a PIL/Pillow image, apply OCR, and then delete
        # # the temporary file
        # # print("C:/Users/mzm/PycharmProjects/My_website/ocr_using_video/"+filename,Image.open("C:\\Users\mzm\PycharmProjects\My_website\ocr_using_video\\"+filename))
        # text = pytesseract.image_to_string(filename)
        # os.remove(filename)
        print("Text in Image :\n",text)

        return jsonify({"text" : text})

app.run("0.0.0.0",5000,threaded=True,debug=True)


