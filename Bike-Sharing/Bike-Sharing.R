# Bike Sharing Demand Prediction / 2021.01.07

library(tidymodels) # machine learning
library(caret)
library(lubridate) # working with date & time
library(tidyverse) # data wrangling
library(moments) # skewness testing
library(corrr) # correlation
bike <- read_csv("Data.csv")
bike %>% dim()
# No variables have missing data.
is.na(bike) %>% colSums()
bike %>% head()

bike %>%
  mutate(instant = NULL, yr = yr + 2011) %>%
  rename(
    date = dteday,
    year = yr,
    month = mnth,
    hour = hr,
    weather = weathersit,
    humidity = hum,
    total = cnt
  ) ->
bike

bike %>%
  pivot_longer(
    cols = c(casual, registered, total),
    names_to = "usertype",
    values_to = "count"
  ) ->
bike_long

# EDA ---------------------------------------------------------------------

# Rental count
bike_long %>%
  ggplot(aes(count, colour = usertype)) +
  geom_density() +
  labs(
    title = "Distribution of the number of rental bikes",
    x = "Number per hour", y = "Density"
  ) +
  scale_colour_discrete(
    name = "User type",
    breaks = c("casual", "registered", "total"),
    labels = c("Non-registered", "Registered", "Total")
  )

# By year
bike_long %>%
  filter(!usertype == "total") %>%
  ggplot(aes(as.factor(year), count)) +
  geom_violin(aes(fill = usertype))

# By holiday
bike %>%
  ggplot(aes(total, colour = as.factor(holiday))) +
  geom_density()

# By working day
bike %>%
  ggplot(aes(total, colour = as.factor(workingday))) +
  geom_density()

# By weather
bike %>%
  ggplot(aes(total, colour = as.factor(weather))) +
  geom_density()

# By temp
bike %>%
  ggplot(aes(temp * 41, total)) +
  geom_point() +
  geom_smooth()

# Data Preparation --------------------------------------------------------

# "casual" and "registered" are excluded
bike_all <- bike %>%
  select(-casual, -registered)

# Correlation
bike_all %>%
  select(where(is.numeric)) %>%
  correlate() %>%
  rearrange(absolute = FALSE) %>%
  shave() ->
bike_cor
rplot(bike_cor, print_cor = TRUE)

# Target variable is checked for skewness and transformed
skewness(bike_all$total)
skewness(log10(bike_all$total))
skewness(log1p(bike_all$total))
skewness(sqrt(bike_all$total))
skewness(bike_all$total^(1 / 3))
bike_all$total <- bike_all$total^(1 / 3)

# Recode variables
bike_all %>% summary()
bike_all$season <- factor(
  bike_all$season,
  levels = c(1, 2, 3, 4),
  labels = c("spring", "summer", "autumn", "winter")
)
bike_all$holiday <- factor(
  bike_all$holiday,
  levels = c(0, 1), labels = c(FALSE, TRUE)
)
bike_all$workingday <- factor(
  bike_all$workingday,
  levels = c(0, 1), labels = c(FALSE, TRUE)
)
bike_all$weather <- factor(
  bike_all$weather,
  levels = c(1, 2, 3, 4),
  labels = c("clear", "cloudy", "rainy", "heavy rain"),
  ordered = TRUE
)

set.seed(25)
split <- initial_split(bike_all, prop = 0.8)
train_data <- training(split)
test_data <- testing(split)
train_cv <- vfold_cv(train_data, v = 10)
train_cv_caret <- rsample2caret(train_cv)

prep_recipe <-
  recipe(total ~ ., data = train_data) %>%
  step_rm(year, month, weekday) %>%
  step_date(date) %>%
  step_corr(all_numeric(), threshold = 0.8) %>%
  step_dummy(all_nominal())

train_data_caret <-
  prep(prep_recipe) %>% bake(new_data = NULL)

test_data_caret <-
  prep(prep_recipe) %>% bake(new_data = test_data)

ctrl_caret <- trainControl(
  method = "cv",
  index = train_cv_caret$index,
  indexOut = train_cv_caret$indexOut
)

# Modelling ---------------------------------------------------------------

# Generate prediction tables
predict_table <- function(model, data, tidy_flag) {
  if (tidy_flag == TRUE) {
    result <- model %>%
      predict(data) %>%
      rename(pred = .pred) %>%
      mutate(
        actual = data$total,
        pred_real = pred^3,
        actual_real = actual^3
      )
  } else {
    result <- model %>%
      predict(data) %>%
      as_tibble_col(column_name = "pred") %>%
      mutate(
        actual = data$total,
        pred_real = pred^3,
        actual_real = actual^3
      )
  }
  result
}

# Extract RMSE for tidymodels
pull_rmse <- function(result_table) {
  rmse_result <- rmse(result_table, pred, actual) %>%
    pull(.estimate)
  rmse_result_real <- rmse(result_table, pred_real, actual_real) %>%
    pull(.estimate)
  result <- c(rmse = rmse_result, real_rmse = rmse_result_real)
}


# Baseline
base_train_pred <-
  tibble(actual = train_data$total, actual_real = train_data$total^3) %>%
  mutate(pred = mean(actual), pred_real = mean(actual_real))
base_test_pred <-
  tibble(actual = test_data$total, actual_real = test_data$total^3) %>%
  mutate(pred = mean(actual), pred_real = mean(actual_real))
base_train_rmse <- pull_rmse(base_train_pred)
print(base_train_rmse)
base_test_rmse <- pull_rmse(base_test_pred)
print(base_test_rmse)


# Decision Tree
tree_cp <- seq(0.01, 0.1, 0.01)

set.seed(25)
tree_caret_time1 <- Sys.time()
tree_caret <- train(
  total ~ .,
  data = train_data_caret,
  method = "rpart",
  trControl = ctrl_caret,
  metric = "RMSE",
  tuneGrid = data.frame(cp = tree_cp)
)
tree_caret_time2 <- Sys.time()
print(tree_caret_time2 - tree_caret_time1)

tree_caret_train_pred <- predict_table(tree_caret, train_data_caret, FALSE)
tree_caret_train_rmse <- pull_rmse(tree_caret_train_pred)
print(tree_caret_train_rmse)
tree_caret_test_pred <- predict_table(tree_caret, test_data_caret, FALSE)
tree_caret_test_rmse <- pull_rmse(tree_caret_test_pred)
print(tree_caret_test_rmse)

set.seed(25)
tree_tidy_time1 <- Sys.time()
tree_engine <-
  decision_tree(mode = "regression", cost_complexity = tune()) %>%
  set_engine("rpart")

tree_workflow <-
  workflow() %>%
  add_recipe(prep_recipe) %>%
  add_model(tree_engine)

tree_tune <- tune_grid(
  tree_workflow,
  resamples = train_cv,
  grid = data.frame(cost_complexity = tree_cp),
  metrics = metric_set(rmse)
)
tree_tidy_time2 <- Sys.time()
print(tree_tidy_time2 - tree_tidy_time1)

tree_best <-
  finalize_workflow(tree_workflow, select_best(tree_tune)) %>%
  fit(train_data)

tree_tidy_train_pred <- predict_table(tree_best, train_data, TRUE)
tree_tidy_train_rmse <- pull_rmse(tree_tidy_train_pred)
print(tree_tidy_train_rmse)
tree_tidy_test_pred <- predict_table(tree_best, test_data, TRUE)
tree_tidy_test_rmse <- pull_rmse(tree_tidy_test_pred)
print(tree_tidy_test_rmse)

rbind(
  base_train_rmse, base_test_rmse,
  tree_tidy_train_rmse, tree_tidy_test_rmse,
  tree_caret_train_rmse, tree_caret_test_rmse
)



# Random forest

set.seed(25)
rf_caret_t1 <- Sys.time()
rf_caret <- train(
  total ~ .,
  data = bake(prep_caret, new_data = NULL),
  method = "rf",
  trControl = ctrl_caret,
  metric = "RMSE",
  importance = TRUE,
  tuneLength = 3
)
rf_caret_t2 <- Sys.time()
print(rf_caret_t2 - rf_caret_t1)

rf_engine <-
  rand_forest(mode = "regression", mtry = tune(), trees = tune()) %>%
  set_engine("randomForest")

rf_grid <- expand.grid(
  mtry = c(10, 20),
  trees = c(100, 500, 1000)
)

rf_workflow <-
  workflow() %>%
  add_recipe(prep_recipe) %>%
  add_model(rf_engine)

rf_tune <-
  tune_grid(
    rf_workflow,
    resamples = train_cv,
    grid = rf_grid,
    metrics = metric_set(rmse)
  )
collect_metrics(rf_tune)

rf_best <-
  finalize_workflow(rf_workflow, select_best(rf_tune)) %>%
  fit(train_data)

rf_train_pred <- predict_table(rf_best, train_data)
rf_test_pred <- predict_table(rf_best, test_data)

rf_train_rmse <- pull_rmse(rf_train_pred)
print(rf_train_rmse)
rf_test_rmse <- pull_rmse(rf_test_pred)
print(rf_test_rmse)



# KNN
str_time <- Sys.time()
knn_engine <-
  nearest_neighbor(mode = "regression", neighbors = tune()) %>%
  set_engine("kknn")

knn_grid <- expand.grid(neighbors = seq(2, 20, 2))

knn_workflow <-
  workflow() %>%
  add_recipe(prep_recipe) %>%
  add_model(knn_engine)

knn_tune <-
  tune_grid(
    knn_workflow,
    resamples = train_cv,
    grid = knn_grid,
    metrics = metric_set(rmse)
  )
end_time <- Sys.time()
print(end_time - str_time)
collect_metrics(knn_tune)

knn_best <-
  finalize_workflow(knn_workflow, select_best(knn_tune)) %>%
  fit(train_data)

knn_train_pred <- predict_table(knn_best, train_data)
knn_train_rmse <- pull_rmse(knn_train_pred)
print(knn_train_rmse)
knn_test_pred <- predict_table(knn_best, test_data)
knn_test_rmse <- pull_rmse(knn_test_pred)
print(knn_test_rmse)
