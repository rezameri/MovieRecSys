########################################################################
################## MOVIE RECOMMENDATION SYSTEM PROJECT ################# 
##########################  Reza Ameri, Ph.D. ########################## 
########################################################################

########################## Description of the Project

#Recommender systems have become prevalent in recent years as they tackle the problem 
#of information overload by suggesting the most relevant products to end users. In fact, 
#recommender systems are information filtering tools that aspire to predict the rating for 
#users and items, predominantly from big data to recommend their likes. Specifically, 
#movie recommendation systems provide a mechanism to assist users in classifying users 
#with similar interests. In this project, exploratory data analysis is used in order to 
#develop various machine learning algorithms that predict movie ratings with reasonable 
#accuracy. The project is part of the capstone for the professional certificate in data 
#science program at Harvard University. The MovieLens 10M Dataset that includes 10 million 
#ratings and 100,000 tag applications applied to 10,000 movies by 72,000 users is used for 
#creating a movie recommendation system.

########################## MovieLens Dataset

# Create train and validation sets

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

########################## Data Processing

# Structure of the dataset
str(edx)

# Headers of the dataset
head(edx) %>%
  print.data.frame()

# Summary of the dataset
summary(edx)

# Ratings per movie distribution 
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25, color="black", fill="blueviolet") +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Movies") +
  ggtitle("Distribution of Movie Ratings") + theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# User ratings distribution 
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25, color="black", fill="blueviolet") +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Users") +
  ggtitle("Distribution of Users Ratings") + theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Star ratings distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color="black", fill="blueviolet") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  xlab("Rating") +
  ylab("Frequency") +
  ggtitle("Distribution of Star Ratings") + theme_bw() + 
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Average movie ratings by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(binwidth = 0.25, color="black", fill="blueviolet") +
  xlab("Average Star Ratings") +
  ylab("Number of Users") +
  ggtitle("Average Movie Ratings by Users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) + theme_bw() + 
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Distribution of rated movies based on the release date
edx$year <- as.numeric(substr(as.character(edx$title),nchar(as.character(edx$title))-4,
                              nchar(as.character(edx$title))-1))

edx %>%
  ggplot(aes(edx$year)) +
  geom_histogram(binwidth = 0.5, color="black", fill="blueviolet") + 
  xlab("Release Date") +
  ylab("Frequency") + 
  ggtitle("Average Rating Based on Release Date") + theme_bw() + 
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

# Average rating based on genre
dat <- edx %>% separate_rows(genres, sep ="\\|")

dat %>%
  group_by(genres) %>%
  summarize(n=n()) %>%
  ungroup() %>%
  mutate(sumN = sum(n), percentage = n/sumN) %>%
  arrange(-percentage) %>% 
  ggplot(aes(reorder(genres, percentage), percentage, fill= percentage)) +
  geom_bar(stat = "identity") + coord_flip() +
  scale_fill_distiller(palette = "Purples") + labs(y = "Percentage", x = "Genre") +
  ggtitle("Average Rating Based on Genre") + theme_bw()

########################## Methodology and Procedure

RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

############## Method I: Average Movie Rating

mu <- mean(edx$rating)
cat("mu = ", mu)

naive_rmse <- RMSE(validation$rating, mu)
cat("RMSE = ", naive_rmse)

rmse_results <- data_frame(Method = "Method I: Average Movie Rating", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

############## Method II:  Movie Effect Model

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_2_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method II:  Movie Effect Model", 
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

############## Method III: Movie & User Effect Model

user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method III: Movie & User Effect Model",  
                                     RMSE = model_3_rmse))
rmse_results %>% knitr::kable()

############## Method IV: Regularized Movie & User Effect Model

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))})

qplot(lambdas, rmses) + xlab("Lambda") + ylab("RMSE") + theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text.y   = element_text(size=11),
        axis.text.x   = element_text(size=11),
        axis.title.y  = element_text(size=14),
        axis.title.x  = element_text(size=14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        panel.background=element_rect(fill = "white"))

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method IV: Regularized Movie & User Effect Model"
                                     , RMSE = min(rmses)))
rmse_results %>% knitr::kable()

############## Method V: Parallel Matrix Factorization Model

edx_factorization <- edx %>% select(movieId, userId, rating)
validation_factorization <- validation %>% select(movieId, userId, rating)

edx_factorization <- as.matrix(edx_factorization)
validation_factorization <- as.matrix(validation_factorization)

write.table(edx_factorization, file = "trainingset.txt", sep = " ", row.names = FALSE, 
            col.names = FALSE)

write.table(validation_factorization, file = "validationset.txt", sep = " ", 
            row.names = FALSE, col.names = FALSE)

set.seed(1)
training_dataset <- data_file("trainingset.txt")

validation_dataset <- data_file("validationset.txt")

r = Reco() # this will create a model object

opts = r$tune(training_dataset, opts = list(dim = c(10, 20, 30), lrate = c(0.1,
    0.2), costp_l1 = 0, costq_l1 = 0, nthread = 1, niter = 10))

r$train(training_dataset, opts = c(opts$min, nthread = 1, niter = 20))
stored_prediction = tempfile() 

r$predict(validation_dataset, out_file(stored_prediction))
real_ratings <- read.table("validationset.txt", header = FALSE, sep = " ")$V3
pred_ratings <- scan(stored_prediction)

model_5_rmse <- RMSE(real_ratings, pred_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Method V: Parallel Matrix Factorization Model", 
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()