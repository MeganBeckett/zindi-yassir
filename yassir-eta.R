# =====================================================================================================================
#
# Yassir ETA prediction - UmojaHack SA
#
# Megan Beckett                                
# https://www.linkedin.com/in/meganbeckett/     
# https://twitter.com/mbeckett_za               
#
# =====================================================================================================================

# This is based on the data from https://zindi.africa/hackathons/umojahack-south-africa-yassir-eta-prediction-challenge/data.

# STAGES --------------------------------------------------------------------------------------------------------------

# 1. Acquire data.
# 2. Load data.
# 3. Wrangle data.
# 4. Exploratory analysis.
# 5. Build models.

# ACQUIRE DATA --------------------------------------------------------------------------------------------------------

# 1. Create an account on Zindi.
# 2. Login to Zindi and join the hackathon.
# 3. Navigate to the URL above.
# 4. Download the data.
# 5. Unpack the ZIP archive into a data/ folder.

# LIBRARIES -----------------------------------------------------------------------------------------------------------
library(Metrics)             # Evaluating model performance
# library(MASS)                # Stepwise Algorithm
library(caret)               # Swiss Army Knife for ML

library(readr)               # Reading CSV files
library(dplyr)               # General data wrangling
library(janitor)             # Cleaning column names
library(ggplot2)             # Wicked plots
library(lubridate)           # Handling date/time data

# NOTE: It's important to load dplyr after MASS, otherwise MASS::select will mask dplyr::select.

# LOAD DATA -----------------------------------------------------------------------------------------------------------
PATH_TRAIN <- file.path("data", "Train.csv")
PATH_TEST <- file.path("data", "Test.csv")
PATH_WEATHER <-file.path("data", "Weather.csv")

# Read in the data.
trips <- read_csv(PATH_TRAIN)
weather <- read_csv(PATH_WEATHER)

# Take a look at the column names.
names(trips)
names(weather)

# Improve column names (using snake case).
trips <- trips %>% clean_names()
names(trips)

# Look at structure of data.
str(trips)

# Look at a "spreadsheet" view of data.
View(trips)

# We can immediately drop the ID column since this cannot have any predictive value.
trips <- trips %>% select(-id)

# WRANGLE -------------------------------------------------------------------------------------------------------------
# Create a date from the timestamp for trips to be able to join the weather data by date
trips <- trips %>%
  mutate(date = as.Date(timestamp))


# FEATURE ENGINEERING -------------------------------------------------------------------------------------------------
# Convert the trip start numeric time of day (perhaps time of day plays a part, for example during peak traffic hours)
trips <- trips %>% 
  mutate(time_of_day = lubridate::hour(timestamp)
         )


# WEATHER DATA --------------------------------------------------------------------------------------------------------
# Add in the weather data for each trip according to its date of departure
trips <- trips %>% 
 left_join(weather, by = "date")


# EDA: PLOTS ----------------------------------------------------------------------------------------------------------
# ETA compared to trip distance - there is a linear relationship
ggplot(trips, aes(x = trip_distance, y = eta)) +
  geom_point(alpha = 0.1) +
  labs(x = "Trip distance (m)", y = "ETA (seconds)")

# Distribution of eta
ggplot(trips, aes(x = eta)) +
  geom_histogram()

# Distribution of trip distance
ggplot(trips, aes(x = trip_distance)) +
  geom_histogram()

# Distribution of rain fall
ggplot(trips, aes(x = total_precipitation)) +
  geom_histogram()

# For interest, let's visualise where these trips are mostly departing from.
# Perhaps there is a city/urban/rural factor to take into account later
# We will plot this with the leaflet library
library(leaflet)

leaflet(trips) %>% 
  addTiles() %>% 
  addMarkers(
    lng = ~origin_lon,
    lat = ~origin_lat,
    clusterOptions = markerClusterOptions()
)


# TRAIN/TEST SPLIT ----------------------------------------------------------------------------------------------------

# In order to assess how well our model performs we need to split it into two components:
#
# - training and
# - testing.
#
# There are a number of ways to do this split.

# Set the RNG seed so that everybody gets the same train/test split.
#
set.seed(13)

# Generally you want to have around 80:20 split.
#
index <- sample(c(TRUE, FALSE), nrow(trips), replace = TRUE, prob = c(0.8, 0.2))

train <- trips[index,]
test <- trips[!index,]

# Check that the proportions are correct (should be roughly 4x as many records in training data).
#
nrow(train)
nrow(test)


# MODEL KNN -----------------------------------------------------------------------------------------------------------
# We’ll kick off by looking at something that is not a linear model: k-Nearest Neighbours (kNN). 
# The principle is simple: assign a value derived from a collection of “nearby” observations.
# This technique is very flexible (it works well for both classification and regression problems).
library(kknn)

trips_knn <- kknn(eta ~ .,
                  train, test, k = 7, kernel = "optimal")

# Calculate the RMSE.
rmse(test$eta, predict(trips_knn))


# MODEL LM ------------------------------------------------------------------------------------------------------------
# We then look at simple linear regression model. And just throw everything in - the "kitchen sink" approach.
trips_lm <- lm(eta ~ trip_distance + time_of_day +
                 # Weather data 
                 dewpoint_2m_temperature + maximum_2m_air_temperature + mean_2m_air_temperature +
                 mean_sea_level_pressure + minimum_2m_air_temperature + surface_pressure +
                 total_precipitation + u_component_of_wind_10m + v_component_of_wind_10m,
                 data = train)

summary(trips_lm)

# What about some interactions between different weather recordings for the day, for example rain and wind?
trips_lm <- update(trips_lm, . ~ . + total_precipitation:u_component_of_wind_10m)
#
summary(trips_lm)

# Make predictions on the testing data.
#
test_predictions <- predict(trips_lm, test)
head(test_predictions)
#
# So how well does the model perform? Need to compare the known to the predicted classes.
#
head(test$eta)
#
# The standard approach to evaluating a linear regression model is to calculate the RMSE. 
# Calculate the RMSE.
rmse(test$eta, test_predictions)

# This is a rather large RMSE. It's also larger than our k-Nearest Neighbours model.




