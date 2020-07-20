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

# library(rpart)               # Decision Tree models
# library(rattle)              # Visualising Decision Tree models (can be a problem to install on Windows)
# library(rpart.plot)          # Visualising Decision Tree models
library(Metrics)             # Evaluating model performance
library(MASS)                # Stepwise Algorithm
library(caret)               # Swiss Army Knife for ML

library(readr)               # Reading CSV files
library(dplyr)               # General data wrangling
library(forcats)             # Working with factors
library(janitor)             # Cleaning column names
library(naniar)              # Missing data
library(ggplot2)             # Wicked plots
library(lubridate)           # Handling date/time data

# NOTE: It's important to load dplyr after MASS, otherwise MASS::select will mask dplyr::select.

# LOAD DATA -----------------------------------------------------------------------------------------------------------

PATH_TRAIN <- file.path("data", "Train.csv")
#
PATH_TEST <- file.path("data", "Test.csv")
#
PATH_WEATHER <-file.path("data", "Weather.csv")

# Read in the data.
#
trips <- read_csv(PATH_TRAIN)
weather <- read_csv(PATH_WEATHER)

# Take a look at the column names.
#
names(trips)
names(weather)

# Improve column names (using snake case).
#
trips <- trips %>% clean_names()
#
names(trips)

# Look at structure of data.
#
str(trips)

# Look at a "spreadsheet" view of data.
#
View(trips)
#
# We can immediately drop the ID column since this cannot have any predictive value.
#
trips <- trips %>% select(-id)

# WRANGLE -------------------------------------------------------------------------------------------------------------
# Create a date from the timestamp for trips to be able to join the weather data by date
trips <- trips %>%
  mutate(date = as.Date(timestamp))

# FEATURE ENGINEERING -------------------------------------------------------------------------------------------------
# Convert the trip start numeric time of day (perhaps time of day plays a part, for example during peak traffic hours)
#
trips <- trips %>% 
  mutate(time_of_day = lubridate::hour(timestamp)
         )

# WEATHER DATA --------------------------------------------------------------------------------
# Add in the weather data for each trip according to its date of departure
#
trips <- trips %>% 
 left_join(weather, by = "date")

# EDA: PLOTS ----------------------------------------------------------------------------------------------------------
library(leaflet)

leaflet

# This EDA has only considered univariate relationships.
#
# It could be extended with a multivariate analysis.

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
index <- sample(c(TRUE, FALSE), nrow(insurance), replace = TRUE, prob = c(0.8, 0.2))

train <- insurance[index,]
test <- insurance[!index,]

# Check that the proportions are correct (should be roughly 4x as many records in training data).
#
nrow(train)
nrow(test)

# What about the proportions of the target variable?
#
prop.table(table(train$car_insurance))
prop.table(table(test$car_insurance))
#
# These look fairly close. Good enough for the moment.

# MODEL #1 ------------------------------------------------------------------------------------------------------------

# A Decision Tree model is a great place to start. It should produce reasonable result and it also gives a model which
# is simple to interpret.

model_rpart <- rpart(car_insurance ~ ., data = train)
#
# This is a "kitchen sink" model: we're throwing in all of the features.
#
# Fortunately a Decision Tree works rather well for this because it'll perform implicit feature selection, retaining
# only those features which do actually contribute.

# Take a quick look at the model.
#
model_rpart
#
# What's the most important feature?

# Plot the tree.
#
fancyRpartPlot(model_rpart)
rpart.plot(model_rpart, cex = 0.75)
#
# Interpretations:
#
# - The longer the call, the more likely the sale.
# - Long calls less effective for elderly. Simply enjoy the chat?

# MODEL ASSESSMENT ----------------------------------------------------------------------------------------------------

# Make predictions on the testing data.
#
test_predictions <- predict(model_rpart, test)
head(test_predictions)
#
# These are the predicted probabilities of the outcome classes.
#
# Can we get a class prediction?
#
?predict.rpart
#
# We need to specify the 'type' parameter.

# Specify that we want to predict classes.
#
test_predictions <- predict(model_rpart, test, type = "class")
head(test_predictions)
#
# Bingo!

# So how well does the model perform? Need to compare the known to the predicted classes.
#
head(test$car_insurance)
#
# Looks reasonable. But let's be more rigorous!

# Compare predictions to known values.
#
test_predictions == test$car_insurance

# What proportion of these are correct?
#
mean(test_predictions == test$car_insurance)

# There's a function for this!
#
accuracy(test$car_insurance, test_predictions)
#
# This is the proportion of predictions that are correct.
#
# There's a problem with this though: if our model is really good at predicting the negative class then it will still
# have good accuracy.

# MODEL #2 ------------------------------------------------------------------------------------------------------------

# Unfortunately our model appears to be better than it actually is.
#
# Why?
#
# Because it's reliant on a feature that we could not have prior to contacting the potential client: call duration!

# 🚨 Exercise
#
# 1. Remove the columns relating to the call.
# 2. Repeat the train/test split.
# 3. Build a Decision Tree model.
# 4. Plot the tree. What's the most important feature?
# 4. Make predictions on the testing set.
# 5. Calculate the accuracy.

# === -> YOUR CODE ===
# Let's create a fmore realisticdataset.
#
# We need to remove features that relate to the call.
#
# Why?
#
# Because we don't know the time or duration of the call in advance. These are not characteristics of the prospective
# customer. So to use these data we are effectively "snooping" the results.
#
insurance <- insurance %>% select(-starts_with("call_"))
#
train <- insurance[index,]
test <- insurance[!index,]

model_rpart <- rpart(car_insurance ~ ., data = train)

test_predictions <- predict(model_rpart, test, type = "class")

accuracy(test$car_insurance, test_predictions)

fancyRpartPlot(model_rpart)
rpart.plot(model_rpart, cex = 0.75)
# === <- YOUR CODE ===

# DAY OF WEEK ---------------------------------------------------------------------------------------------------------

# 🚨 Exercise (BONUS)
#
# We have the month and day of last contact. Is it possible to guess the year?
#
# It's possible that not all of the contacts happened in the same year, but you should be able to get an idea of a
# likely year.

# === -> YOUR CODE ===
# Strategy: Guess a year and then find the day of week corresponding to the specified month and day. Find a year which
# minimises the number of calls on Saturday and Sunday.

YEAR <- 2015

insurance %>%
  mutate(
    last_contact_date = as.Date(sprintf("2015-%s-%d", last_contact_month, last_contact_day), format = "%Y-%b-%d"),
    last_contact_weekday = wday(last_contact_date, label = TRUE)
  ) %>%
  select(starts_with("last_contact")) %>%
  count(last_contact_weekday)
# === <- YOUR CODE ===

# MODEL #3: LOGISTIC REGRESSION ---------------------------------------------------------------------------------------

# 📌 Logistic Function
#
tibble(
  x = seq(-15, 15, 0.25),
  y = plogis(x)
) %>% ggplot(aes(x, y)) + geom_line()
#
# A Logistic Regression model uses a logistic "link function" to map the real numbers onto the interval [0, 1].

model_glm <- glm(car_insurance ~ ., data = train, family = binomial)

summary(model_glm)
#
# All of the features get coefficients, many of which are not statistically significant.

# Let's take a look at how this model performs.
#
test_predictions <- predict(model_glm, test, type = "response")
head(test_predictions)
#
# These are effectively probabilities. We need to apply a threshold to convert them to classes.
#
test_predictions <- ifelse(test_predictions > 0.5, 1, 0)

accuracy(test$car_insurance, test_predictions)
#
# The performance of the Logistic Regression model is similar to that of the Decision Tree.

# MODEL #4 ------------------------------------------------------------------------------------------------------------

# We could narrow down the selection of coefficients manually, but we can also automate the process.
#
model_glm <- stepAIC(model_glm)

# What terms are there in the new model?
#
model_glm$formula
#
# We've dropped these predictors:
#
# - job
# - marital
# - default
# - balance
# - last_contact_day and
# - days_passed.
#
# So the new model is considerably more parsimonious.

# What effect has this had on model performance?
#
test_predictions <- predict(model_glm, test, type = "response")
test_predictions <- ifelse(test_predictions > 0.5, 1, 0)
#
accuracy(test$car_insurance, test_predictions)
#
# It's not quite as good as the model with all of the terms, but the difference is really very small (and could
# change with a different train/test split).

# MODEL #5: CARET / DECISION TREE -------------------------------------------------------------------------------------

# Let's take the model up a notch and use caret.

# 📌 Testing, Validation & Cross-Validation
#
# https://raw.githubusercontent.com/datawookie/useful-images/master/data-train-test.svg
# https://raw.githubusercontent.com/datawookie/useful-images/master/data-train-test-validate.svg
# https://raw.githubusercontent.com/datawookie/useful-images/master/cross-validation.svg

# First we need to make a small change to the target variable because target expects the positive class to be the first
# level.
#
insurance$car_insurance = relevel(insurance$car_insurance, "1")
levels(insurance$car_insurance) <- c("yes", "no")
#
train <- insurance[index,]
test <- insurance[!index,]

# In caret models are created with train().
#
# What models are possible?
#
names(getModelInfo())

model_rpart <- train(car_insurance ~ ., data = train, method = "rpart")

model_rpart
#
# The accuracy estimate is far more robust because it's been calculated with boostrapping rather than a single split.

# Generate predictions on the testing data.
#
test_predictions <- predict(model_rpart, test)

confusionMatrix(test_predictions, test$car_insurance)
#
# Now we have access to a whole suite of metrics (in addition to the accuracy):
#
# sensitivity - what proportion of the positive values are correctly predicted [*]
# specificity - what proportion of the negative values are correctly predicted
#
# positive predictive value - what proportion of the positive predictions are correct [*]
# negative predictive value - what proportion of the negative predictions are correct

# Can we use predictions to rate prospective clients?
#
# Get the predicted probability of a successful call.
#
test_probabilities <- predict(model_rpart, test, type = "prob")$yes
#
test %>%
  mutate(rating = test_probabilities) %>%
  arrange(desc(rating))

# What are the most important predictors?
#
varImp(model_rpart)

# MODEL #6: CARET / XGBOOST -------------------------------------------------------------------------------------------

TRAINCONTROL = trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

model_xgboost <- train(
  car_insurance ~ .,
  data = train,
  method = "xgbTree",
  # Optimise the model for sensitivity.
  metric = "Sens",
  trControl = TRAINCONTROL
)

test_predictions <- predict(model_xgboost, test)

confusionMatrix(test_predictions, test$car_insurance)
#
# Finally a model that's better than random guessing! Compare the model sensitivity to the 40% chance of guessing
# correctly.

test %>%
  mutate(rating = predict(model_xgboost, test, type = "prob")$yes) %>%
  arrange(desc(rating))

varImp(model_xgboost)