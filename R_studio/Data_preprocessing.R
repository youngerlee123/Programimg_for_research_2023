require(rio)
require(dplyr)
setwd('C:/Users/User/Desktop/dataset/R studio')
MYCN = import('../0502-Kelly-MYCN/combined_protein2.tsv')
IGG = import('../0502-Kelly-IGG/combined_protein1.tsv')

MYCNCTP <- MYCN$`Combined Total Peptides`
IGGCTP <- IGG$`Combined Total Peptides`

MYCNCTSC <- MYCN$`Combined Total Spectral Count`
IGGCTSC <- IGG$`Combined Total Spectral Count`

MYCNIntensity1 <- MYCN$`1_1 Intensity`
MYCNIntensity2 <- MYCN$`1_2 Intensity`
MYCNIntensity3 <- MYCN$`1_3 Intensity`

IGGIntensity1 <- IGG$`1_1 Intensity`
IGGIntensity2 <- IGG$`1_2 Intensity`
IGGIntensity3 <- IGG$`1_3 Intensity`

MYCNNAME <- MYCN$`Entry Name`
IGGNAME <- IGG$`Entry Name`


i=1
dname=0
dpep=0
d1=0
d1_data<-data.frame(Gene=(1:2100),MYCNpeptide=(1:2100),IGGpeptide=(1:2100),MYCN_Spectral_Count=(1:2100),IGG_Spectral_Count=(1:2100),MYCNInt1=(1:2100),MYCNInt2=(1:2100),MYCNInt3=(1:2100),IGGInt1=(1:2100),IGGInt2=(1:2100),IGGInt3=(1:2100))
while (MYCNNAME[i] !='NA') {
  g=1
  while (IGGNAME[g] !='NA'){
    if(IGGNAME[g] == MYCNNAME[i]){
      d1_data$Gene[i] <- MYCNNAME[i]
      d1_data$MYCNpeptide[i] <- MYCNCTP[i]
      d1_data$IGGpeptide[i] <-IGGCTP[g]
      d1_data$MYCN_Spectral_Count[i] <- MYCNCTSC[i]
      d1_data$IGG_Spectral_Count[i] <- MYCNCTSC[i]
      d1_data$MYCNInt1[i] <- MYCNIntensity1[i]
      d1_data$MYCNInt2[i] <- MYCNIntensity2[i]
      d1_data$MYCNInt3[i] <- MYCNIntensity3[i]
      d1_data$IGGInt1[i] <- IGGIntensity1[g]
      d1_data$IGGInt2[i] <- IGGIntensity2[g]
      d1_data$IGGInt3[i] <- IGGIntensity3[g]
      if(d1_data$MYCNInt1[i]==0){
        d1_data$MYCNInt1[i]=NA
      }
      if(d1_data$MYCNInt2[i]==0){
        d1_data$MYCNInt2[i]=NA
        
      }
      if(d1_data$MYCNInt3[i]==0){
        d1_data$MYCNInt3[i]=NA
        
      }
      if(d1_data$IGGInt1[i]==0){
        d1_data$IGGInt1[i]=NA
        
        
      }
      if(d1_data$IGGInt2[i]==0){
        d1_data$IGGInt2[i]=NA
        
        
      }
      if(d1_data$IGGInt3[i]==0){
        d1_data$IGGInt3[i]=NA
     
        
        
      }
      
      break
    }
    if(g==1726) {
      d1_data$Gene[i] <- NA
      d1_data$MYCNpeptide[i] <- NA
      d1_data$IGGpeptide[i] <-NA
      d1_data$MYCN_Spectral_Count[i] <- NA
      d1_data$IGG_Spectral_Count[i] <- NA
      d1_data$MYCNInt1[i] <- NA
      d1_data$MYCNInt2[i] <- NA
      d1_data$MYCNInt3[i] <-NA
      d1_data$IGGInt1[i] <- NA
      d1_data$IGGInt2[i] <- NA
      d1_data$IGGInt3[i] <- NA
      
      break}
    g=g+1
    
  }  
  if(i==2026){break}
  i=i+1
}
d2 <- na.omit(d1_data)  #remove NA
d1_data <-d1_data[which(rowSums(d1_data==0)==0),]
write.csv(d2, file="protein比較0504-2.csv", row.names = FALSE)
