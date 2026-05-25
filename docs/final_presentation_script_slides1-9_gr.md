# Σενάριο Τελικής Παρουσίασης — Διαφάνειες 1–9
**EPL 445 | Final Presentation | Andreas Demosthenous & Marios Olympios | 25 Μαΐου 2026**

> Χρόνος ομιλίας 1–9: περίπου 9 με 10 λεπτά.

---

## Διαφάνεια 1 — Τίτλος

Καλησπέρα. Ανδρέας Δημοσθένους και Μάριος Ολύμπιος, EPL 445. Το project λέγεται *MobileViT for Real-Time Traffic Monitoring*.

Φτιάξαμε ένα ολοκληρωμένο σύστημα που ταξινομεί, ανιχνεύει και κάνει tracking οχήματα μέσα από browser. Το μοντέλο MobileViT-S έχει 5,6 εκατομμύρια παραμέτρους. Πετυχαίνουμε 96,44% test accuracy. Η pipeline τρέχει σε integrated GPU, χωρίς dedicated κάρτα: YOLOv8-nano, MobileViT-S, SORT και FastAPI.

---

## Διαφάνεια 2 — Agenda

Η ροή σήμερα έχει οχτώ στάδια. Πρώτα το πρόβλημα και γιατί επιλέξαμε αυτή την κατεύθυνση. Μετά το dataset και το pre-processing, όπου κάναμε μια διόρθωση που άλλαξε τα νούμερα από fake σε αληθινά. Συνεχίζουμε με την αρχιτεκτονική MobileViT και τη διαδικασία εκπαίδευσης.

Στο τέταρτο μέρος εξηγούμε γιατί αφήσαμε το sliding window του Interim 2 και περάσαμε σε υβριδικό σχήμα με YOLO. Πέμπτο, το tracking με SORT και η καταμέτρηση ανά λωρίδα. Έκτο, το web dashboard. Τελευταία τα ποσοτικά αποτελέσματα και ένα live demo.

---

## Διαφάνεια 3 — Πρόβλημα και Κίνητρο

Το ερευνητικό ερώτημα: φτάνει ένα ελαφρύ vision transformer για να τροφοδοτήσει ένα live, web-based dashboard καταμέτρησης ανά λωρίδα, τρέχοντας σε integrated graphics;

Τα digital-twin συστήματα για έξυπνες πόλεις δουλεύουν με live counts. Δεν τους αρκεί batch processing που κρατάει ώρες. Επιλέξαμε το MobileViT γιατί 5,6 εκατομμύρια παράμετροι χωράνε άνετα στα 8 GB shared VRAM της Radeon 780M. Ένα ViT-base δεν χωράει.

Σχεδιαστικά, αποφύγαμε το end-to-end YOLO. Κρατήσαμε το MobileViT για ταξινόμηση και βάλαμε το YOLOv8-nano μπροστά μόνο για να προτείνει σφιχτά boxes. Έτσι κρατάμε τη δική μας 4-class ταξονομία: car, bus, truck, background. Και κρατάμε όλη την εκπαίδευση που έχουμε ήδη κάνει.

Όλη η εφαρμογή σηκώνεται με ένα `uvicorn` command. Webcam ή RTSP πάει σε YOLO, μετά σε MobileViT, μετά σε SORT, και βγαίνει στο dashboard.

---

## Διαφάνεια 4 — Dataset και Pre-processing

Πηγή είναι το UA-DETRAC από το Kaggle mirror. Έχει 140.000 annotated frames από 100 surveillance sequences. Από εκεί κόβουμε crops κάθε bounding box που ξεπερνά τα 32×32 pixels. Βάλαμε ανώτατο όριο 5.000 crops ανά κλάση για να μην έχουμε class imbalance. Συνολικά καταλήγουμε στα 475.000 patches.

Το label mapping το προσαρμόσαμε: `car → car`, `bus → bus`, `van/others → truck`. Τα background crops τα φτιάξαμε εμείς, παίρνοντας δείγματα από τις γωνίες κάθε frame.

Στο pre-processing κάνουμε resize στα 256×256, normalization με ImageNet statistics επειδή το backbone είναι pretrained σε ImageNet, και augmentations: horizontal flip, ±10° rotation, color jitter.

Εδώ θέλω να σταθώ στο leakage fix. Στα αρχικά splits, που ήταν τυχαία ανά patch, frames από το ίδιο video sequence εμφανίζονταν τόσο στο train όσο και στο test. Πετυχαίναμε ~99% accuracy. Ήταν fake νούμερο. Σπάσαμε τα splits ανά sequence ID, δηλαδή ολόκληρα videos μπαίνουν εξ ολοκλήρου σε train, val ή test. Η ακρίβεια έπεσε στο 96,44%, που είναι το πραγματικό νούμερο.

---

## Διαφάνεια 5 — Αρχιτεκτονική MobileViT-S

Οι Mehta και Rastegari πρότειναν το MobileViT στο ICLR 2022. Συνδυάζει τοπική συνέλιξη για textures, edges και χρώματα με global self-attention για context μέσα σε κάθε εικόνα.

Ξεκινάει με ένα 3×3 conv που κατεβάζει την ανάλυση στα 128×128. Ακολουθούν inverted residual blocks με depthwise convolutions από το MobileNetV2. Μετά έρχονται δύο MobileViT blocks: το πρώτο έχει 192 channels και 4 transformer layers, το δεύτερο έχει 240 channels και 3 transformer layers. Κλείνει με global average pooling και ένα linear layer στις 4 κλάσεις.

Όλο μαζί κάνει 5,6 εκατομμύρια παραμέτρους, 4 φορές μικρότερο από ResNet-50 με συγκρίσιμη ακρίβεια στο UA-DETRAC. Φορτώνουμε pretrained weights από το `timm` και κάνουμε fine-tuning για 15 epochs.

Το πρακτικό κέρδος: χωράει στα 8 GB shared VRAM της Radeon 780M και τρέχει με batch size 4. Ένα ViT-base δεν θα έτρεχε καθόλου.

---

## Διαφάνεια 6 — Διαδικασία Εκπαίδευσης

Κάνουμε δι-φασικό fine-tuning. Στα πρώτα 3 epochs παγώνουμε το backbone και εκπαιδεύεται μόνο το classification head. Έτσι το head μαθαίνει τις δικές μας κλάσεις χωρίς να καταστρέφει τα pretrained weights. Από το epoch 4 ξεπαγώνουμε το backbone και ρίχνουμε το learning rate 10 φορές, ώστε τα pretrained features να προσαρμοστούν αργά στα δεδομένα κυκλοφορίας.

Χρησιμοποιούμε AdamW με αρχικό learning rate 3e-4 και weight decay 1e-4. Δουλεύει καλύτερα από SGD σε fine-tuning transformers. Το schedule είναι cosine annealing για 15 epochs.

Το loss είναι class-weighted cross-entropy. Τα background και bus samples ήταν υποεκπροσωπευμένα και θέλαμε να αντισταθμίσουμε.

Στο hardware: AMD Radeon 780M iGPU με ROCm 7.2. Το gfx1103 δεν υπάρχει στο rocBLAS, οπότε κάνουμε spoofing με `HSA_OVERRIDE_GFX_VERSION=11.0.0`. Το batch size κάθεται στα 4. Πιο πάνω κρασάρει ο compositor όταν ξεπαγώνει το backbone.

Έχουμε early stopping στο macro F1 με patience 5. Το καλύτερο checkpoint βγήκε στο epoch 13, με validation macro F1 0,9807.

---

## Διαφάνεια 7 — Pipeline Ανίχνευσης

Εδώ φαίνεται η μεγαλύτερη αλλαγή από το Interim 2. Είχαμε sliding window baseline. Τρία μεγέθη παραθύρου: 120, 180, 256 px, με stride 50%. Κάθε tile το ταξινομούσε το MobileViT, και μετά κάναμε per-class greedy NMS με IoU 0,3. Δούλευε, αλλά είχε δύο πρακτικά προβλήματα. Πρώτο, πολλά overlapping detections. Δεύτερο, χιλιάδες background tiles που έπρεπε να ταξινομηθούν χωρίς να βγάλουμε τίποτα από αυτά. Έτρεχε στα 3 fps σε 640×360 frames.

Η νέα προσέγγιση είναι υβριδική. Το YOLOv8-nano προτείνει ένα σφιχτό box ανά όχημα. Κρατάμε μόνο τις COCO κλάσεις 2 (car), 3 (motorcycle), 5 (bus) και 7 (truck). Κάθε crop περνάει μετά από το MobileViT για ταξινόμηση στις δικές μας 4 κλάσεις. Όσα boxes πάρουν label `background`, τα πετάμε.

Φτάσαμε στα 4 fps steady-state, στο ίδιο hardware, με ένα μόνο box ανά αντικείμενο και πολύ λιγότερα duplicates. Το σημαντικό: δεν χρειάστηκε να ξανατρέξουμε εκπαίδευση. Το ίδιο MobileViT, διαφορετικός proposer.

---

## Διαφάνεια 8 — Cross-Frame Tracking με SORT

SORT των Bewley et al., ICIP 2016. Είναι ένας ελαφρύς tracker που δουλεύει μόνο με IoU association, χωρίς appearance embeddings.

Πώς λειτουργεί. Σε κάθε frame, οι ανιχνεύσεις γίνονται match με τα ενεργά trackers με βάση το IoU. Όσες δεν matchάρουν, δημιουργούν νέο tracker. Όσα trackers δεν επιβεβαιωθούν για 5 frames, αποσύρονται. Το `max_age=5` δίνει αντοχή σε σύντομα occlusions.

Έχουμε επίσης `min_hits=2`. Ένα track χρειάζεται 2 διαδοχικά confirmations πριν εμφανιστεί στο output. Έτσι σκοτώνουμε τα μονο-frame false positives πριν φτάσουν στους counters.

Κάθε γραμμή στο CSV παίρνει πλέον `track_id`. Στη συνάθροιση βγάζουμε δύο νούμερα. Το raw detection count, και το `unique_vehicles_by_class`. Το raw νούμερο υπερτιμά τη ροή γιατί κάθε όχημα εμφανίζεται σε δεκάδες frames. Το SORT μας δίνει πόσα μοναδικά οχήματα πέρασαν από την κάμερα. Αυτό θέλει το digital twin.

---

## Διαφάνεια 9 — Per-Lane Counting και Web Dashboard

Η καταμέτρηση ανά λωρίδα γίνεται post-hoc, χωρίς re-inference. Ο χειριστής σχεδιάζει κλειστά polygons πάνω στο πρώτο frame, μέσα από HTML5 canvas. Στο backend, για κάθε ανίχνευση ελέγχουμε αν το κέντρο του bounding box βρίσκεται μέσα σε κάθε polygon, με `cv2.pointPolygonTest`. Δεν τρέχει ξανά το μοντέλο. Διαβάζουμε το CSV που έχει ήδη γραφτεί. Ο χειριστής μπορεί να αλλάξει τις λωρίδες ή να προσθέσει νέες με μηδέν κόστος.

Διαλέξαμε FastAPI για το backend. Τα endpoints είναι λίγα και ξεκάθαρα: `POST /jobs` για upload, `GET /jobs/{id}/counts | /timeline | /video | /frame`, και `POST /jobs/{id}/lanes` και `/roi` για τις γεωμετρίες. Το FastAPI μας δίνει BackgroundTasks για το job queue και auto-generated OpenAPI στο `/docs`.

Το frontend κρατήθηκε απλό. Ένα index.html, ένα app.js, ένα style.css. Καθόλου build step, καθόλου bundlers. Chart.js από CDN για τα bar και line charts. HTML5 canvas για το ROI και lane editing.

Ένα τελευταίο σημείο που μας έφαγε χρόνο. Το OpenCV γράφει videos σε mp4v codec, που οι browsers δεν παίζουν. Φτιάξαμε ένα `/video` endpoint που κάνει lazy re-encode σε libx264 / yuv420p με `+faststart`. Ο worker κάνει pre-warm μετά από κάθε job, οπότε όταν ο χρήστης πατήσει play, το video ξεκινάει αμέσως.

---

*Σύνολο 9 με 10 λεπτά για τις διαφάνειες 1–9. Οι 10–18 (αποτελέσματα, demo, conclusions, Q&A) έρχονται σε επόμενο μέρος.*
