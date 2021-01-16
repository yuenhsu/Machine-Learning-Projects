---
title: "Bike Sharing Demand Prediction"
subtitle: "Creating machine learning models with caret and tidymodels"
author: "Yu-En Hsu"
date: "2021-01-16"
output: 
  html_document: 
    keep_md: yes
    toc: yes
    toc_depth: 2
    code_folding: show
---



## Overview

I demonstrated how to create predictive models with `caret` and `tidymodels`. The data are from [Bike Sharing Dataset from UCI repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

### Data

Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic, environmental and health issues.

Apart from interesting real world applications of bike sharing systems, the characteristics of data being generated by these systems make them attractive for the research. Opposed to other transport services such as bus or subway, the duration of travel, departure and arrival position is explicitly recorded in these systems. This feature turns bike sharing system into a virtual sensor network that can be used for sensing mobility in the city. Hence, it is expected that most of important events in the city could be detected via monitoring these data.

### Attributes

- instant: record index
- dteday : date
- season : season (1:winter, 2:spring, 3:summer, 4:fall)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- hr : hour (0 to 23)
- holiday : weather day is holiday or not (extracted from [Web Link])
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
+ weathersit :
  - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
  - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered

***

## Import data


```r
library(tidymodels) 
library(caret)
library(lubridate) 
library(tidyverse) 
library(moments) 
library(corrr) 
library(randomForest)
bike <- read_csv("Data.csv")
bike %>% dim()
## [1] 17379    17
```

### Organise variables

I removed `instant`, changed the formatting for `year`, and renamed some variables. Because the data are in wide format, in which `casual`, `registered`, and `total` are in three separate columns, I pivoted the three variables to long format data frame, `bike_long`.


```r
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
head(bike)
```

```
## # A tibble: 6 x 16
##   date       season  year month  hour holiday weekday workingday weather  temp
##   <date>      <dbl> <dbl> <dbl> <dbl>   <dbl>   <dbl>      <dbl>   <dbl> <dbl>
## 1 2011-01-01      1  2011     1     0       0       6          0       1  0.24
## 2 2011-01-01      1  2011     1     1       0       6          0       1  0.22
## 3 2011-01-01      1  2011     1     2       0       6          0       1  0.22
## 4 2011-01-01      1  2011     1     3       0       6          0       1  0.24
## 5 2011-01-01      1  2011     1     4       0       6          0       1  0.24
## 6 2011-01-01      1  2011     1     5       0       6          0       2  0.24
## # … with 6 more variables: atemp <dbl>, humidity <dbl>, windspeed <dbl>,
## #   casual <dbl>, registered <dbl>, total <dbl>
```


```r
bike %>%
  pivot_longer(
    cols = c(casual, registered, total),
    names_to = "usertype",
    values_to = "count"
  ) ->
bike_long
head(bike_long)
```

```
## # A tibble: 6 x 15
##   date       season  year month  hour holiday weekday workingday weather  temp
##   <date>      <dbl> <dbl> <dbl> <dbl>   <dbl>   <dbl>      <dbl>   <dbl> <dbl>
## 1 2011-01-01      1  2011     1     0       0       6          0       1  0.24
## 2 2011-01-01      1  2011     1     0       0       6          0       1  0.24
## 3 2011-01-01      1  2011     1     0       0       6          0       1  0.24
## 4 2011-01-01      1  2011     1     1       0       6          0       1  0.22
## 5 2011-01-01      1  2011     1     1       0       6          0       1  0.22
## 6 2011-01-01      1  2011     1     1       0       6          0       1  0.22
## # … with 5 more variables: atemp <dbl>, humidity <dbl>, windspeed <dbl>,
## #   usertype <chr>, count <dbl>
```

***

## Explore data {.tabset}

Here I provided some quick ways to visualise the distributions and relationships among variables. 

### Target variable

The distributions of rental counts are positively skewed. It is desirable to have normal distribution as most machine learning techniques require the dependent variable to be normal. I addressed the skewness later in data engineering.


```r
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
```

![](Bike-Sharing_files/figure-html/plot_target-1.png)<!-- -->

### By year


```r
bike_long %>%
  filter(!usertype == "total") %>%
  ggplot(aes(as.factor(year), count)) +
  geom_violin(aes(fill = usertype))
```

![](Bike-Sharing_files/figure-html/plot_year-1.png)<!-- -->

### By holiday


```r
bike %>%
  ggplot(aes(total, colour = as.factor(holiday))) +
  geom_density()
```

![](Bike-Sharing_files/figure-html/plot_holiday-1.png)<!-- -->

### By working day


```r
bike %>%
  ggplot(aes(total, colour = as.factor(workingday))) +
  geom_density()
```

![](Bike-Sharing_files/figure-html/plot_working-1.png)<!-- -->

### By weather


```r
bike %>%
  ggplot(aes(total, colour = as.factor(weather))) +
  geom_density()
```

![](Bike-Sharing_files/figure-html/plot_weather-1.png)<!-- -->

### By temperature


```r
bike %>%
  ggplot(aes(temp * 41, total)) +
  geom_point() +
  geom_smooth()
```

![](Bike-Sharing_files/figure-html/plot_temp-1.png)<!-- -->

### Correlation


```r
bike %>%
  select(where(is.numeric)) %>%
  correlate() -> bike_cor
rplot(bike_cor, print_cor = TRUE)
```

![](Bike-Sharing_files/figure-html/plot_cor-1.png)<!-- -->

***

## Prepare data

### Target variable

I focused on the `total` count, so `casual` and `registered` variables are moved. As suggested earlier, the target variable is positively skewed and requires transformation. I tried several common techniques for dealing with positively skewed data and applied the one with the lowest skewness.


```r
bike_all <- bike %>%
  select(-casual, -registered)

# Original
skewness(bike_all$total)
## [1] 1.277301

# Log
skewness(log10(bike_all$total))
## [1] -0.936101

# Log + constant
skewness(log1p(bike_all$total))
## [1] -0.8181098

# Square root
skewness(sqrt(bike_all$total))
## [1] 0.2864499

# Cubic root
skewness(bike_all$total^(1 / 3))
## [1] -0.0831688

# Transform with cubic root
bike_all$total <- bike_all$total^(1 / 3)
```

### Predictors

Categorical variables are converted to factors according to the attribute information provided by UCI.


```r
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
head(bike_all)
```

```
## # A tibble: 6 x 14
##   date       season  year month  hour holiday weekday workingday weather  temp
##   <date>     <fct>  <dbl> <dbl> <dbl> <fct>     <dbl> <fct>      <ord>   <dbl>
## 1 2011-01-01 spring  2011     1     0 FALSE         6 FALSE      clear    0.24
## 2 2011-01-01 spring  2011     1     1 FALSE         6 FALSE      clear    0.22
## 3 2011-01-01 spring  2011     1     2 FALSE         6 FALSE      clear    0.22
## 4 2011-01-01 spring  2011     1     3 FALSE         6 FALSE      clear    0.24
## 5 2011-01-01 spring  2011     1     4 FALSE         6 FALSE      clear    0.24
## 6 2011-01-01 spring  2011     1     5 FALSE         6 FALSE      cloudy   0.24
## # … with 4 more variables: atemp <dbl>, humidity <dbl>, windspeed <dbl>,
## #   total <dbl>
```

***

## Split data

Earlier, I was just cleaning and exploring the data. While I transformed some variables, only the transformation applicable to the entire dataset was done. Starting from here, the code involves `caret` and `tidymodels`, and the differences between these libraries kick in. I don't know whether it's because Max Kuhn built both packages or whether `tidymodels` accommodate large number of `caret` users, there are functions allowing me to use two packages seamlessly. 

### tidymodels

The tidymodels `rsample` library handles data splitting. Training and testing split is done as shown, along with a 10-fold cross-validation.


```r
set.seed(25)
split <- initial_split(bike_all, prop = 0.8)
train_data <- training(split)
train_data %>% dim()
## [1] 13904    14

test_data <- testing(split)
test_data %>% dim()
## [1] 3475   14

train_cv <- vfold_cv(train_data, v = 10)
```

### caret

There are two options available:

1. Use caret's `createDataPartition` and other functions to split data and specify cross-validation method. The code is hidden here:


```{.r .fold-hide}
set.seed(25)
train_index <- createDataPartition(
  bike_all$total, p = 0.8, times = 1, list = FALSE
)
train_data <- mics[ train_index, ]
test_data  <- mics[-train_index, ]

fold_index <- createFolds(
  train_data$total,
  k = 10, returnTrain = TRUE, list = TRUE
)
train_cv <- trainControl(method="cv", index = fold_index)
```

2. Use tidymodel's `rsample2caret` function, which returns a list that mimics the `index` and `indexOut` elements of a trainControl object.


```r
train_cv_caret <- rsample2caret(train_cv)
ctrl_caret <- trainControl(
  method = "cv",
  index = train_cv_caret$index,
  indexOut = train_cv_caret$indexOut
)
```

***

## Preprocess data

### tidymodels

The `recipe` is used for preprocessing. Typically, it's followed by `prep()` and `bake()` to transform the data. However, as I set up a workflow later, I only specified the recipe.


```r
prep_recipe <-
  recipe(total ~ ., data = train_data) %>%
  step_rm(year, month, weekday) %>%
  step_date(date) %>%
  step_corr(all_numeric(), threshold = 0.8) %>%
  step_dummy(all_nominal())
```

### caret

Similarly, I can use caret's `preProcess()` function. But I always find it frustrating because all numerical variables are processed, and there is not much flexibility. 


```{.r .fold-hide}
prep <- preProcess(cutoff = 0.8)
```

Again, tidymodels' `recipe` can be used for caret. Here, I `prep()` and `bake()` to transform the data because caret does not have a workflow function.


```r
train_data_caret <-
  prep(prep_recipe) %>% bake(new_data = NULL)

test_data_caret <-
  prep(prep_recipe) %>% bake(new_data = test_data)
```

***

## Model


```{.r .fold-hide}
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
```

### Baseline

Two functions are hidden above. The baseline is the average of the `total`.


```r
base_train_pred <-
  tibble(actual = train_data$total, actual_real = train_data$total^3) %>%
  mutate(pred = mean(actual), pred_real = mean(actual_real))
base_test_pred <-
  tibble(actual = test_data$total, actual_real = test_data$total^3) %>%
  mutate(pred = mean(actual), pred_real = mean(actual_real))
base_train_rmse <- pull_rmse(base_train_pred)
print(base_train_rmse)
```

```
##       rmse  real_rmse 
##   2.032927 181.063306
```

```r
base_test_rmse <- pull_rmse(base_test_pred)
print(base_test_rmse)
```

```
##      rmse real_rmse 
##   2.02608 182.61370
```

### Decision tree

#### tidymodels

`parsnip` for modelling, `workflow` for well... workflow, `tune` for parameter tuning, and `yardstick` for performance metrics. I was also curious about the timing, so I recorded the time as well. 


```r
# Cost complexity for decision tree parameter
tree_cp <- seq(0.01, 0.1, 0.01)

set.seed(25)
tree_tidy_time1 <- Sys.time()

# Specify model
tree_engine <- 
  decision_tree(mode = "regression", cost_complexity = tune()) %>%
  set_engine("rpart")

# Set workflow (Preprocess & model)
tree_workflow <-
  workflow() %>%
  add_recipe(prep_recipe) %>% 
  add_model(tree_engine)

# Tune parameters with cross-validation
tree_tune <- tune_grid(
  tree_workflow,
  resamples = train_cv,
  grid = data.frame(cost_complexity = tree_cp),
  metrics = metric_set(rmse)
)

# Fit again with the best parameter
tree_best <-
  finalize_workflow(tree_workflow, select_best(tree_tune)) %>%
  fit(train_data)

tree_tidy_time2 <- Sys.time()
print(tree_tidy_time2 - tree_tidy_time1)
```

```
## Time difference of 1.33398 mins
```

```r
tree_tidy_train_pred <- predict_table(tree_best, train_data, TRUE)
tree_tidy_train_rmse <- pull_rmse(tree_tidy_train_pred)
print(tree_tidy_train_rmse)
```

```
##       rmse  real_rmse 
##   1.078724 116.106006
```

```r
tree_tidy_test_pred <- predict_table(tree_best, test_data, TRUE)
tree_tidy_test_rmse <- pull_rmse(tree_tidy_test_pred)
print(tree_tidy_test_rmse)
```

```
##       rmse  real_rmse 
##   1.074347 118.205989
```

#### caret


```r
set.seed(25)
tree_caret_time1 <- Sys.time()
tree_caret <- train(
  total~.,
  data = train_data_caret,
  method = "rpart",
  trControl = ctrl_caret,
  metric = "RMSE",
  tuneGrid = data.frame(cp = tree_cp)
)
tree_caret_time2 <- Sys.time()
print(tree_caret_time2 - tree_caret_time1)
```

```
## Time difference of 4.455021 secs
```

```r
tree_caret_train_pred <- predict_table(tree_caret, train_data_caret, FALSE)
tree_caret_train_rmse <- pull_rmse(tree_caret_train_pred)
print(tree_caret_train_rmse)
```

```
##       rmse  real_rmse 
##   1.078724 116.106006
```

```r
tree_caret_test_pred <- predict_table(tree_caret, test_data_caret, FALSE)
tree_caret_test_rmse <- pull_rmse(tree_caret_test_pred)
print(tree_caret_test_rmse)
```

```
##       rmse  real_rmse 
##   1.074347 118.205989
```

### Comparison


```r
rbind(
  base_train_rmse, base_test_rmse,
  tree_tidy_train_rmse, tree_tidy_test_rmse,
  tree_caret_train_rmse, tree_caret_test_rmse
)
```

```
##                           rmse real_rmse
## base_train_rmse       2.032927  181.0633
## base_test_rmse        2.026080  182.6137
## tree_tidy_train_rmse  1.078724  116.1060
## tree_tidy_test_rmse   1.074347  118.2060
## tree_caret_train_rmse 1.078724  116.1060
## tree_caret_test_rmse  1.074347  118.2060
```


As you can see, the results for the decision tree model are the same regardless of the library, since I split the data and set up cross-validation the same way. Moreover, both tidymodels and caret use `rpart` as the underlying engine. 

I have been using tidymodels for a few weeks now, and I really enjoy the integration to the tidyverse. But I find it confusing to have so many steps and objects. For example, I keep trying to get RMSE from a workflow object. I probably just need a bit more time to get familiar with the new ecosystem.

One thing I notice is how much longer tidymodel takes to tune the parameters than caret does. For the decision tree, tidymodels takes over 1 minute while caret only needs 4-5 seconds. 

