# 讀取文件
setwd('C:/Users/User/Desktop/dataset/R studio')
data <- read.csv("MYCN火山圖測試.csv")

library(ggplot2)
library(ggrepel)  #用於標記

FC = 1.5 # 用於判斷上下調的foldchange
PValue = 0.05 
a=-log10(PValue)

# 添加sig列用於標示上調/下調
data$sig[(data$PV < -1*log10(PValue)|data$PV=="NA")|(data$FC < log2(FC))& data$FC > -log2(FC)] <- "NotSig"
data$sig[data$PV >= -1*log10(PValue) & data$FC >= log2(FC)] <- "Up"
data$sig[data$PV >= -1*log10(PValue) & data$FC <= -log2(FC)] <- "Down"

#選擇FC與PValue超過多少的蛋白質要標示出名字
 PvalueLimit = 15
 FCLimit = 15
 data$label=ifelse(data$PV > PvalueLimit & data$FC >= FCLimit, as.character(data$Entry_Name), '')

# 繪圖
p.vol<-ggplot(data,aes(data$FC,data$PV)) +    
  geom_point(aes(color = sig)) +                           # 繪製火山圖，設定圖片的XY軸界線)
  scale_x_continuous(limits = c(-10, 10)) +
  scale_y_continuous(limits = c(0, 15)) +
  labs(title="volcanoplot",                                # 設定XY軸名稱與標題
       x="log[2](FC)", 
       y="-log[10](PValue)") + 
  # scale_color_manual(values = c("red","green","blue")) + # 可自訂義顏色
  geom_hline(yintercept=-log10(PValue),linetype=2)+        
  geom_vline(xintercept=c(-log2(FC),log2(FC)),linetype=2)+ 
  geom_text_repel(aes(x =data$FC,                   
                      y = data$PV,          
                      label=label),                       
                  max.overlaps = 10000,                    # 最大覆蓋率，當點很多時，有些標記會被覆蓋，調大該值則不被覆蓋。
                  size=3,                                  # 字體大小
                  box.padding=unit(0.5,'lines'),           # 標記蛋白質的邊距
                  point.padding=unit(0.1, 'lines'), 
                  segment.color='black',                   # 標記線條的顏色
                  show.legend=FALSE)

ggsave(p.vol,filename = "Volcano0425 MYC.pdf")


