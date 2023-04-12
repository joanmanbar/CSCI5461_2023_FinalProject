# Count data

# Set wd (if necessary) 
setwd(dirname(rstudioapi::getSourceEditorContext()$path))




# *****************************************************
# Libraries ----
# *****************************************************

library(tidyr)
# install.packages("devtools")  # install 'devtools' in R(>3.0.2)
# devtools::install_github('gangwug/MetaCycle') # install MetaCycle
library(MetaCycle)




# *****************************************************
# Files ----
# *****************************************************

TPM_counts <- read.delim('../input/tpm_counts.txt') # counts
TPM <- TPM_counts # Copy original data
rownames(TPM) <- TPM$Geneid  # Genes as index
TPM <- TPM[,-1]  # Remove Geneid column
# Read Key file
Key <- read.delim('../input/LIBRARIES_KEY.txt', sep="\t", header = F)
# Add proper headers
Key_headers <- c('libraryName','SampleId','rawReads','filteredReads','sampleName','conditionNumber','groupName','sequencerType','runType','fileUsed')
colnames(Key) <- Key_headers  
# Match library names with actual sample id
match_index <- match(names(TPM), Key$libraryName)
colnames(TPM) <- Key$sampleName[match_index]
# We only need WTP1 to WTP6
cols_to_keep <- grep("WTP", colnames(TPM)) # Select only WTP
TPM <- TPM[, cols_to_keep] # Subset
# ********** WHAT IS T1?? ********************
cols_to_remove <- grepl("WTP7|WTP8|_T1", names(TPM)) # Remove WTP7 and WTP8 
TPM <- TPM[, !cols_to_remove] # subset
# Get info from sample ID
IDs <- data.frame(names(TPM))
names(IDs) <- "ID"
split_ID <- separate(IDs, col = ID, 
                       into = c("Genotype", "WTP", "Rep"), sep = "_")
IDs <- cbind(IDs, split_ID)




# *****************************************************
# Missing samples ----
# *****************************************************

genotypes <- sort(unique(IDs$Genotype)) # Unique genotypes
time_points <- sort(unique(IDs$WTP)) # Unique time points
reps <- sort(unique(IDs$Rep)) # Unique reps
n_genotypes <- length(genotypes) # total genotypes
n_reps <- length(reps) # total time points
n_tp <- length(time_points) # total reps
genotypes <- unlist(c(lapply(genotypes, rep, times = n_reps*n_tp))) # all genotypes
time_points <- unlist(c(lapply(time_points, rep, times = n_reps))) # tps by reps
time_points <- rep(time_points, n_genotypes) # all time points
reps <- rep(reps, n_tp*n_genotypes) # all reps
# Get all combinations in order
all_samples <- unlist(Map(paste, genotypes, time_points, reps, sep = "_"))
all_samples <- unname(all_samples)
# identify missing samples
missing_samples <- setdiff(all_samples, names(TPM)) #Difference
# Create a new data frame with NAs for missing samples
NA_df <- data.frame(matrix(NA, nrow = nrow(TPM), ncol = length(missing_samples)))
colnames(NA_df) <- missing_samples # Add colnames




# *****************************************************
# Complete samples ----
# *****************************************************
# Combine original and Na dfs
complete_samples <- cbind(TPM, NA_df)
complete_samples <- complete_samples[all_samples] # order the columns




# *****************************************************
# MetaCycle Analysis----
# *****************************************************
# Source: https://cran.r-project.org/web/packages/MetaCycle/MetaCycle.pdf

# Subset a genotype
AO3 <- complete_samples[,1:24]
AO3$GeneID <- rownames(AO3) # Add rownames as gene Id
AO3 <- AO3[,c(25,1:24)] # bring ID to front

# write file into a 'txt' file
write.table(AO3, file="../input/complete_samples.txt",
            sep="\t", quote=FALSE, row.names=FALSE)

# analyze data with JTK_CYCLE and Lomb-Scargle
meta2d(infile="../input/complete_samples.txt", filestyle="txt", 
       outdir="../output", timepoints=rep(seq(0, 30, by=6), each=4),
       cycMethod=c("JTK","LS"), outIntegration="noIntegration")

# Files should be saved in "output" directory





# *****************************************************
# Ignore this----
# *****************************************************

# This is how example data from MetaCycle look like
df <- cycMouseLiverProtein
df <- cycHumanBloodData
df2 <- cycHumanBloodDesign
df <- cycYeastCycle
