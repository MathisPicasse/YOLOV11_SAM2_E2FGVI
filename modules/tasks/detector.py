from ultralytics import YOLO

class Detector: 
    def __init__(self, model:str, tracker: str):
        self.model = YOLO(model)
        self.tracker = tracker
        self.raw_results = None

    def track(self, video): 
        self.raw_results = self.model.track(source=video, tracker=self.tracker, show=False)
        return self.raw_results

    def results_format(self):
        results_format = {}
        # raw_results est une liste d'objets Results, un par frame
        for frame_id, res in enumerate(self.raw_results):
            # res.boxes.id et res.boxes.xyxy sont des tensors PyTorch
            ids = res.boxes.id.cpu().numpy().astype(int)  # convertir en array numpy int
            boxes = res.boxes.xyxy.cpu().numpy()          # array Nx4
            
            for obj_id, box in zip(ids, boxes):
                if obj_id in results_format:
                    results_format[obj_id].append((frame_id, box))
                else:
                    results_format[obj_id] = [(frame_id, box)]
        return results_format

