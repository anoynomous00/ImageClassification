import io
import unittest
from PIL import Image
from app import app

class TestClassify(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_classify_prompts(self):
        # Create a dummy red image
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        prompts = "This is an image of a red square.\nThis is an image of a blue circle."
        
        data = {
            'image': (img_byte_arr, 'test.png'),
            'prompts': prompts
        }

        response = self.app.post('/classify', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        
        json_data = response.get_json()
        print("\nResponse:", json_data)
        
        self.assertIn('prediction', json_data)
        self.assertIn('probabilities', json_data)
        self.assertIn('explanation', json_data)
        
        # The red square prompt should have higher probability
        probs = json_data['probabilities']
        red_prob = next(p['probability'] for p in probs if 'red square' in p['label'])
        blue_prob = next(p['probability'] for p in probs if 'blue circle' in p['label'])
        
        self.assertGreater(red_prob, blue_prob)
        self.assertIn("red square", json_data['prediction'])

if __name__ == '__main__':
    unittest.main()
