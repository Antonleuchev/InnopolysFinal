class Result:
    def __init__(self, original_path, processed_path, date, time):
        self.original_path = original_path
        self.processed_path = processed_path
        self.date = date
        self.time = time
        
class ResultWeb:
    def __init__(self, original_img_byte, processed_img_byte, date, time):
        self.original_img_byte = original_img_byte
        self.processed_img_byte = processed_img_byte
        self.date = date
        self.time = time