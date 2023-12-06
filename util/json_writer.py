import json

class Writer():
    def __init__(self):
        self.f = None
        self._init_buffer()

    def _init_buffer(self):
        self.buffer = {
            "metadata": {},
            "annotations": []
        }

    def _buffer_metadata(self, dataset_name):
        from datetime import datetime
        dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.buffer["metadata"] = {"dataset_name": dataset_name, "creation_date": dt}

    def begin(self, filename, dataset_name = "Training set keypoints"):
        # init metadata
        self._buffer_metadata(dataset_name)
        self.f = open(filename, "w")
    
    def buffer_data(self, image_id, keypoints):
        self.buffer["annotations"].append({"image_id": image_id, "keypoints": keypoints})

    def is_open(self):
        return self.f and not self.f.closed
    
    def close(self):
        assert self.f
        json.dump(self.buffer, self.f)
        self.f.close()
        self._init_buffer()
