from app import app
import numpy as np
from PIL import Image
import io

client = app.test_client()

img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)

with client.session_transaction() as sess:
    sess['role'] = 'admin'

try:
    response = client.post('/admin_detect', data={'file': (img_byte_arr, 'test.jpg')})
    print("STATUS CODE:", response.status_code)
    if response.status_code == 500:
        print("Data:", response.data.decode())
except Exception as e:
    import traceback
    traceback.print_exc()
