import cv2
import requests
from flask import Flask, request
from flask_restful import Resource, Api

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


app = Flask(__name__)
api = Api(app)


class PeopleCounterStatic(Resource):
    def get(self):
        # load image
        image = cv2.imread('pap_20230719_1DE.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        return {'peopleCount': len(rects)}

class PeopleCounterDynamicUrl(Resource):
    def get(self):
        url = request.args.get('url')

        response = requests.get(url)

        with open('zdjecie.jpg', 'wb') as file:
	        file.write(response.content)
        print('url', url)

        url = cv2.imread('zdjecie.jpg')

        (rects, weights) = hog.detectMultiScale(url, winStride=(4, 4), padding=(8, 8), scale=1.05)
        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamicUrl, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)
