import requests, json, time, base64

url = "http://localhost:8181/test"
image = "test.jpeg"
image = "multiface_1.jpg"
t1 = time.time()
result = requests.post(url, json={"detect_img": base64.b64encode(open(image, "rb").read())}).text
print(json.loads(result))
print("Cost time: ", time.time()-t1)