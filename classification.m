% 2013-11-06

clear,clc;
database='../data/handwritten/BDL';
classfier='CRC';
nPC=16;

classify_PCA(database,classfier,nPC); 
classify_2DPCA(database,classfier,nPC);   
classify_A2DPCA(database,classfier,nPC);  
classify_2D2PCA(database,classfier,nPC);  
classify_DB2DPCA(database,classfier,nPC); 
