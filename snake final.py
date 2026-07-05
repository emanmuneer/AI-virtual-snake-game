import math
import random
import cvzone
import cv2
import numpy as np
import time
import os
from cvzone.HandTrackingModule import HandDetector

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# ---------------- Rainbow Color Helper ----------------
def getRainbowColor(i,total):
    """Return a BGR rainbow color for snake tail."""
    r = int((math.sin(i/total*math.pi*2)+1)*127)
    g = int((math.sin(i/total*math.pi*2+2)+1)*127)
    b = int((math.sin(i/total*math.pi*2+4)+1)*127)
    return (b,g,r)

# ---------------- Snake Game Class ----------------
class SnakeGame:
    def __init__(self,f,b,p,s):
        # --- Snake State ---
        self.points=[[640,360]]
        self.lengths=[]
        self.currentLength=0
        self.allowedLength=150
        self.previousHead=(640,360)

        # --- Load Images ---
        self.foodImg=cv2.resize(cv2.imread(f,cv2.IMREAD_UNCHANGED),(50,50))
        self.bonusImg=cv2.resize(cv2.imread(b,cv2.IMREAD_UNCHANGED),(50,50))
        self.poisonImg=cv2.resize(cv2.imread(p,cv2.IMREAD_UNCHANGED),(50,50))
        self.superImg=cv2.resize(cv2.imread(s,cv2.IMREAD_UNCHANGED),(60,60))
        self.hFood,self.wFood=self.foodImg.shape[:2]
        self.foodPoint=(0,0)
        self.foodType='normal'
        self.foodTypes=['normal','bonus','poison','super']

        # --- Game State ---
        self.score=0
        self.gameOver=False
        self.level=1
        self.speed=1.0
        self.obstacles=[]
        self.particles=[]
        self.foodTimer=time.time()

        # --- Highscore File ---
        self.highscoreFile="highscore.txt"
        if os.path.exists(self.highscoreFile):
            with open(self.highscoreFile,"r") as f: self.highscore=int(f.read())
        else:
            self.highscore=0

        # --- Initialize Food and Obstacles ---
        self.randomFoodLocation()
        self.generateObstacles()

    # ---------------- Random Food ----------------
    def randomFoodLocation(self):
        self.foodPoint=random.randint(100,1100),random.randint(100,600)
        if self.level%3==0:
            self.foodType=random.choices(['normal','bonus','poison','super'],weights=[65,20,10,5])[0]
        else:
            self.foodType=random.choices(['normal','bonus','poison'],weights=[70,20,10])[0]
        self.foodTimer=time.time()

    # ---------------- Obstacles ----------------
    def generateObstacles(self):
        self.obstacles=[]
        numObs=5+self.level
        for _ in range(numObs):
            x=random.randint(200,1000)
            y=random.randint(100,600)
            size=random.randint(40,80)
            direction=random.choice([-1,1])
            speed=random.uniform(0.5,1.5)
            self.obstacles.append([x,y,size,direction,speed])

    # ---------------- Update Game ----------------
    def update(self,imgMain,currentHead):
        # --- Game Over Display ---
        if self.gameOver:
            cvzone.putTextRect(imgMain,"GAME OVER",[300,300],scale=7,thickness=5,offset=20,colorR=(255,0,0))
            cvzone.putTextRect(imgMain,f"Score:{self.score}",[300,450],scale=5,thickness=4,offset=15)
            cvzone.putTextRect(imgMain,f"High Score:{self.highscore}",[300,550],scale=5,thickness=4,offset=15)
            # Save highscore
            with open(self.highscoreFile,"w") as f: f.write(str(self.highscore))
            return imgMain

        # --- Smooth Head Movement ---
        px,py=self.previousHead
        cx,cy=currentHead
        cx=int(px+(cx-px)*0.15)
        cy=int(py+(cy-py)*0.15)
        self.previousHead=(cx,cy)

        # --- Update Snake ---
        self.points.append([cx,cy])
        d=math.hypot(cx-px,cy-py)
        self.lengths.append(d)
        self.currentLength+=d
        while self.currentLength>self.allowedLength:
            self.currentLength-=self.lengths.pop(0)
            self.points.pop(0)

        # --- Check Food Collision ---
        fx,fy=self.foodPoint
        if fx-self.wFood//2<cx<fx+self.wFood//2 and fy-self.hFood//2<cy<fy+self.hFood//2:
            if self.foodType=='normal': self.allowedLength+=50; self.score+=1
            elif self.foodType=='bonus': self.allowedLength+=100; self.score+=5
            elif self.foodType=='poison': self.allowedLength=max(50,self.allowedLength-50); self.score=max(0,self.score-2)
            elif self.foodType=='super': self.allowedLength+=150; self.score+=10
            for _ in range(15): self.particles.append([cx,cy,random.randint(3,6),random.uniform(-2,2),random.uniform(-2,2),255])
            self.randomFoodLocation()

        # --- Timed Bonus/Poison/Super ---
        if self.foodType in ['bonus','poison','super'] and time.time()-self.foodTimer>5:
            self.randomFoodLocation()

        # --- Level Up ---
        if self.score//10+1>self.level:
            self.level+=1
            self.speed+=0.2
            self.generateObstacles()
        if self.score>self.highscore:
            self.highscore=self.score

        # --- Draw Snake ---
        for i in range(1,len(self.points)):
            cv2.line(imgMain,self.points[i-1],self.points[i],getRainbowColor(i,len(self.points)),20)
        cv2.circle(imgMain,self.points[-1],20,(200,0,200),cv2.FILLED)

        # --- Draw Food ---
        if self.foodType=='normal':
            imgMain=cvzone.overlayPNG(imgMain,self.foodImg,(fx-self.wFood//2,fy-self.hFood//2))
        elif self.foodType=='bonus':
            imgMain=cvzone.overlayPNG(imgMain,self.bonusImg,(fx-self.wFood//2,fy-self.hFood//2))
        elif self.foodType=='poison':
            imgMain=cvzone.overlayPNG(imgMain,self.poisonImg,(fx-self.wFood//2,fy-self.hFood//2))
        else:
            imgMain=cvzone.overlayPNG(imgMain,self.superImg,(fx-self.wFood//2,fy-self.hFood//2))

        # --- Draw Obstacles ---
        for obs in self.obstacles:
            x,y,size,direction,speed=obs
            x+=direction*speed
            if x<0 or x+size>1280:
                obs[3]*=-1
            obs[0]=x
            cv2.rectangle(imgMain,(int(x),int(y)),(int(x)+size,int(y)+size),(0,0,255),cv2.FILLED)

        # --- Draw Scores ---
        cvzone.putTextRect(imgMain,f'Score:{self.score}',[50,50],scale=3,thickness=3,offset=10)
        cvzone.putTextRect(imgMain,f'High Score:{self.highscore}',[50,100],scale=3,thickness=3,offset=10)
        cvzone.putTextRect(imgMain,f'Level:{self.level}',[50,150],scale=3,thickness=3,offset=10)

        # --- Collision Detection ---
        if len(self.points)>20:
            pts=np.array(self.points[:-10],np.int32).reshape((-1,1,2))
            if cv2.pointPolygonTest(pts,(cx,cy),True)>=-1:
                self.gameOver=True
        for obs in self.obstacles:
            x,y,size,_,_=obs
        if x<cx<x+size and y<cy<y+size:
            self.gameOver=True

        # --- Draw Particles ---
        newParticles=[]
        for px,py,size,vx,vy,alpha in self.particles:
            px+=vx
            py+=vy
            alpha-=10
            if alpha>0:
                cv2.circle(imgMain,(int(px),int(py)),size,(255,255,0,int(alpha)),cv2.FILLED); newParticles.append([px,py,size,vx,vy,alpha])
        self.particles=newParticles
        return imgMain

# ---------------- Run Game ----------------
game=SnakeGame("Donut.png","Bonus (1).png","Poison (1).png","Super (1).png")
while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    hands,img=detector.findHands(img,flipType=False)
    if hands:
        img=game.update(img,hands[0]['lmList'][8][0:2])
    cv2.imshow("Snake Game",img)
    key=cv2.waitKey(int(50/game.speed))
    if key==ord('r'):
        game.__init__("Donut.png","Bonus (1).png","Poison (1).png","Super (1).png")
