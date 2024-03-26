# DIM-YOLO: Desktop Item Management System Based on YOLOv8

## Background and Purpose:
Have you ever experienced the frustration of not being able to find items on your desk or in your room because they are misplaced? It would be so convenient if there were a program that, upon telling it what item you're looking for, could tell you where that item is located. Based on this need, we believe an automatic item tracking and management system is necessary. We aim to develop a program capable of automatically tracking identifiable items and their locations within the camera's view, recording them for future queries regarding the item's history and location.

## Implementation Principle:
We developed two programs. The video processing program is responsible for identifying new items, automatically assigning numbers and categories to them, tracking moving items, and recording the status of disappeared items. The program keeps an item log and supports various query operations for items.

The video processing program, named "item_tracker," implements its logic in three main steps:

0. **Import Video** (can be through a built-in computer camera, an external camera, or a video file).
1. **For each frame**, use YOLOv8 to identify each item and its location. Different models can be chosen to prioritize the recognition of certain types of items. The identified items list is recorded as `item_new`.
2. **Read from the database** the list of items still present, recorded as `item_0`, and determine:
    - If an item in `item_new` is within a certain distance of a similar item in the same location in `item_0`, it is considered the same item. Movement is determined based on the distance moved. If there are other items within a certain range, the movement is attributed to that item. This process is repeated for each item in `item_new`.
    - The remaining items in `item_new` are considered to be new or possibly returned items. The history is checked for any disappeared items near the same location. If found, they are considered returned items; if not, they are new items. If there are other items within a certain range, the addition or return is attributed to those items.
    - The remaining items in `item_0` are considered disappeared. If there are other items within a certain range, the disappearance is attributed to obstruction by those items.

3. **Based on these determinations**, the program writes to the log, recording each item's status changes.

The query function is implemented through another piece of code named "search_item":
- This code enables query functionality through searching and modifying the log, supporting queries by item ID, type, appearance time, and tags. The query program also supports modifying item tags.

## Model
For the YOLOv8 model, please refer to the Release on the right side of the page, download it, and put it in the `./model` folder.
