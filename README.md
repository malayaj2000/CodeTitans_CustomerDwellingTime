# Team : CodeTitans
## Project : _CustomerDwellingTime_
  ![python](https://img.shields.io/badge/python%20-3.8-blue)
  ![pytorch](https://img.shields.io/badge/pytorch-1.9-red)
  ![opencv](https://img.shields.io/badge/OpenCV-4.0-green)
  
  Calculation of realtime customer dwelling time using deepSort and YoloV5.
  
  Dwelling Time : Amount of time a person spent in a particular section of a shop/mall
  
### Description
#### Tech Stack : Python , OpenCV ,YoloV5 , Pytorch
  1. Video frames are feed to the deepSort and YoloV5 to obtain the Bounding Box for each person detected in the frame.
  2. Each frame was segmented into 8 Section .
  3. Using Basic geometrical Algorithm we determine the amount of time a person spent,entered,exited a particular section of the shop  
  4. Record the person id ,time spent ,entry time ,exit time ,section in a csv file.
### Motivation/Porpose 
  1. This will help the shopkeeper to keep a record of the products which are sold more so that they can maintain their stock.
  2. Arrange the product in particular fashion to increase their sells.
  3. Automate product ordering in case of festive shopping seasons.  
    

# ToDo
- [ ] Development of UI
- [ ] Impoving Tracking algorithm
- [ ] Developing Embedded model 
