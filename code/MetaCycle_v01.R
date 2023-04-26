# Count data

# Set wd (if necessary) 
setwd(dirname(rstudioapi::getSourceEditorContext()$path))




# *****************************************************
# Libraries ----
# *****************************************************

library(tidyr)
library(reshape2)
library(dplyr)
# install.packages("devtools")  # install 'devtools' in R(>3.0.2)
# devtools::install_github('gangwug/MetaCycle') # install MetaCycle
library(MetaCycle)




# *****************************************************
# Files ----
# *****************************************************

## Match  ----
# Match sample id with library id
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

## Subset ----
# Keep WTP1 to WTP7 for seven specific genotypes
cols_to_keep <- grep("WTP", colnames(TPM)) # Select only WTP
TPM <- TPM[, cols_to_keep] # Subset
cols_to_keep <- grepl("L58|WO83|A03|VT123|Pcglu|O302V|R500", names(TPM)) # Keep 7 genoypes
TPM <- TPM[, cols_to_keep] # Subset
cols_to_remove <- grepl("WTP8|_T1", names(TPM)) # Remove WTP8 and _T1 
TPM <- TPM[, !cols_to_remove] # subset

## Design ----
# Get info from sample ID
IDs <- data.frame(names(TPM))
names(IDs) <- "ID"
split_ID <- separate(IDs, col = ID, 
                       into = c("Genotype", "WTP", "Rep"), sep = "_")
IDs <- cbind(IDs, split_ID)

## Missing samples ----
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
# MetaCycle ----
# *****************************************************
# Source: https://cran.r-project.org/web/packages/MetaCycle/MetaCycle.pdf
# Analysis need to be done per genotype

## All Genotypes ----

### Prep ----
# Timepoints
time_points <- c(17,21,25,29,33,37,41)
time_points <- unlist(c(lapply(time_points, rep, times = n_reps))) # tps by reps
# Genotype
genotypes <- sort(unique(IDs$Genotype)) # Unique


### Analysis ----
# Approx 8.5min per genotype (Joan's Mac)
for (g in 1:length(genotypes)) {
  # Copy dataframe
  df <- complete_samples
  geno_to_keep <- grep(genotypes[g], colnames(df)) # Select specific genotype
  df <- df[, geno_to_keep] # Subset
  df$GeneID <- rownames(df) # Add rownames as gene Id
  df <- df[,c(length(df),1:length(df)-1)] # bring ID to front
  
  # write file into a 'txt' file for this genotype
  df_filename <- sprintf("%s", genotypes[g])
  df_filename <- paste0("../input/MetaCycle/",df_filename)
  if(!file.exists(df_filename)){dir.create(df_filename, recursive = TRUE)} # Veryfy directory exists
  df_filename <- paste0(df_filename,"/input.txt")
  # set utput directory for this genotype
  df_outdir <- sprintf("%s", genotypes[g])
  df_outdir <- paste0("../output/MetaCycle/",df_outdir)
  if(!file.exists(df_outdir)){dir.create(df_outdir, recursive = TRUE)} # Veryfy directory exists
  # Write current genotype's input
  write.table(df, file=df_filename,
              sep="\t", quote=FALSE, row.names=FALSE)
  
  # analyze data with JTK_CYCLE and Lomb-Scargle
  meta2d(infile=df_filename, filestyle="txt", 
         outdir=df_outdir, timepoints=time_points,
         cycMethod=c("JTK"))
  
  print(sprintf("Finished analysis for %s", genotypes[g]))
  
}





# Circadian Genes ----

## Match genes ----
# List synthenic files
Synthenic <- list.files(path = '../input/JGI_syntenicHits', 
                        pattern = '.synHits', 
                        recursive = TRUE, 
                        full.names = TRUE)
# Emtpy df for matching names
MatchGeneName <- data.frame()

for (s in Synthenic) {
  s_file <- read.delim(s) # counts
  s_file <- s_file[, c('id1','id2')]
  colnames(s_file) <- c('CycID','NewCycID')
  colnames(s_file)[colnames(s_file) == 'id1'] <- 'CycID'
  MatchGeneName <- rbind(MatchGeneName, s_file)
}

# Add genotype as column
MatchGeneName <- separate(MatchGeneName, NewCycID, into = c("Genotype", "Gene"), sep = "\\.") 
# Recover NewCycID
MatchGeneName$NewCycID <- paste0(MatchGeneName$Genotype,".",MatchGeneName$Gene)
# Remove the first two characters (Br) in genotype
MatchGeneName$Genotype <- substr(MatchGeneName$Genotype, 
                                 3, 
                                 nchar(MatchGeneName$Genotype))
# Avoid using the function below in case other genotypes contain "Br"
# MatchGeneName$Gene <- gsub('Br','',MatchGeneName$Genotype)


# List files from Metacycle (MC)
MC_files <- list.files(path = '../output/MetaCycle', 
                         pattern = 'meta2d_input.txt', 
                         recursive = TRUE, 
                         full.names = TRUE)

# Create empty list for neew dfs
AllGenotypes <- list()

# Loop through paths
for (f in MC_files) {
  # current genotype
  file_path <- f # path
  GenotypeName <- strsplit(file_path, "/")[[1]]
  GenotypeName <- GenotypeName[length(GenotypeName)-1]
  # Rename WO83, O302V, and Pcglu to match
  GenotypeName <- ifelse(GenotypeName=="WO83","WO_83", 
                         ifelse(GenotypeName=="O302V","O_302V", 
                                ifelse(GenotypeName=="Pcglu","PCGlu",
                                       GenotypeName)))
  
  # Read and filter file
  GenotypeFile <- read.delim(file_path) # counts
  GenotypeFile$CycID <- gsub(pattern = ".v2.1", "", GenotypeFile$CycID)
  GenotypeFile <- GenotypeFile[GenotypeFile$JTK_BH.Q < 0.001, ] # keep those with FDR < 0.001
  
  # We don't need to match R500 since it already has the proper gene names
  if (GenotypeName!="R500") {
    MGN_subset <- MatchGeneName[which(
      MatchGeneName$Genotype ==GenotypeName ), ]
    MGN_subset <- MGN_subset[,c("CycID","NewCycID")]
    # Match gene names
    GenotypeFile <- merge(GenotypeFile,MGN_subset)
    AllGenotypes[[GenotypeName]] <- GenotypeFile  # append
    
    # Only for R500
    } else{ 
      # Add NewCycID column to match the other dfs
      GenotypeFile$NewCycID <- GenotypeFile$CycID 
      AllGenotypes[[GenotypeName]] <- GenotypeFile # append
      
    }
    
}


# Current path
Combined_dfs <- bind_rows(AllGenotypes, .id = "Genotype")
table(Combined_dfs$Genotype)

# ************************************
#     WE HAVE DUPLICATES IN A03 
#         (and maybe others)
#             ASK ANGIE!!!
# ***********************************


# Remove duplicated??????  and save file
CircadianGenes <- Combined_dfs[!duplicated(Combined_dfs$NewCycID), ]
write.csv(CircadianGenes, "../output/MetaCycle/CircadianGenes.csv",row.names = F)


















# ***** ----
# ***** ----
# Ignore   ----
# anything----
# below ----
# ***** ----
# ***** ----







## Plot ----
# 
# # select columns that contain the substring
# WTP1 <- names(df)[grepl(substr_to_find[1], names(df))]
# WTP2 <- names(df)[grepl(substr_to_find[2], names(df))]
# WTP3 <- names(df)[grepl(substr_to_find[3], names(df))]
# WTP4 <- names(df)[grepl(substr_to_find[4], names(df))]
# WTP5 <- names(df)[grepl(substr_to_find[5], names(df))]
# WTP6 <- names(df)[grepl(substr_to_find[6], names(df))]
# WTP7 <- names(df)[grepl(substr_to_find[7], names(df))]
# 
# df2 <- df
# df2$WTP1 <- rowMeans(df[, WTP1], na.rm = TRUE)
# df2$WTP2 <- rowMeans(df[, WTP2], na.rm = TRUE)
# df2$WTP3 <- rowMeans(df[, WTP3], na.rm = TRUE)
# df2$WTP4 <- rowMeans(df[, WTP4], na.rm = TRUE)
# df2$WTP5 <- rowMeans(df[, WTP5], na.rm = TRUE)
# df2$WTP6 <- rowMeans(df[, WTP6], na.rm = TRUE)
# df2$WTP7 <- rowMeans(df[, WTP7], na.rm = TRUE)
# 
# df3 <- df2[,substr_to_find]
# df3$ID <- rownames(df3)
# 
# somegenes <-  c(34829,34923,35159,35240,35255)
# 
# df3 <- df3[somegenes,]
# CirGene <- df3
# CirGene = melt(CirGene, id.vars = "ID")
# 
# ggplot(CirGene, aes(x = variable, y = value)) + geom_line(aes(color = ID, group = ID))
# 
# 
# 
# 
# 
# 
# 
# df3 <- df3[,]
# 
# Top10_A03 <- A03[order(A03$JTK_BH.Q),1]
# Top10_A03 <- Top10_A03[1:5]
# 
# 
# CirGene <- df3[Top10_A03,]
# CirGene$ID <- rownames(CirGene)
# CirGene = melt(CirGene, id.vars = "ID")
# 
# ggplot(CirGene, aes(x = variable, y = value)) + geom_line(aes(color = ID, group = ID))
# 












# ## Single Genotype ----

# ### Prep ----
# Timepoints
# time_points <- c(17,21,25,29,33,37,41)
# time_points <- unlist(c(lapply(time_points, rep, times = n_reps))) # tps by reps
# # Copy df
# df <- complete_samples
# # Subset genotype
# genotypes <- sort(unique(IDs$Genotype)) # Unique genotypes
# g = 1  # specific genotype
# geno_to_keep <- grep(genotypes[g], colnames(df)) # Select specific genotype
# df <- df[, geno_to_keep] # Subset
# df$GeneID <- rownames(df) # Add rownames as gene Id
# df <- df[,c(length(df),1:length(df)-1)] # bring ID to front
# 
# ### Analysis ----
# # write file into a 'txt' file for this genotype
# df_filename <- sprintf("%s", genotypes[g])
# df_filename <- paste0("../input/MetaCycle/",df_filename)
# if(!file.exists(df_filename)){dir.create(df_filename, recursive = TRUE)} # Veryfy directory exists
# df_filename <- paste0(df_filename,"/input.txt")
# # set utput directory for this genotype
# df_outdir <- sprintf("%s", genotypes[g])
# df_outdir <- paste0("../output/MetaCycle/",df_outdir)
# if(!file.exists(df_outdir)){dir.create(df_outdir, recursive = TRUE)} # Veryfy directory exists
# # Write current genotype's input
# write.table(df, file=df_filename,
#             sep="\t", quote=FALSE, row.names=FALSE)
# 
# # analyze data with JTK_CYCLE and Lomb-Scargle
# meta2d(infile=df_filename, filestyle="txt", 
#        outdir=df_outdir, timepoints=time_points,
#        cycMethod=c("JTK","LS"), outIntegration="noIntegration")
# 
# # Files should be saved in "output" directory







# This is how example data from MetaCycle look like
# df <- cycMouseLiverProtein
# df <- cycHumanBloodData
# df2 <- cycHumanBloodDesign
# df <- cycYeastCycle
