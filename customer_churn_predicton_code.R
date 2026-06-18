# ============================================================
# Customer Churn Prediction Using Machine Learning
# Author: Nguyen Tran Thanh Long
# Description:
#   This script builds and compares machine learning models for
#   customer churn prediction using the Telco Customer Churn dataset.
#   The workflow includes data preprocessing, train-test split,
#   one-hot encoding, repeated cross-validation, up-sampling,
#   model comparison, variable importance, and threshold optimization.
# ============================================================


# ============================================================
# 1. Required Packages
# ============================================================

# Run this line once if packages are not installed:
# install.packages(c("caret", "pROC", "randomForest", "rpart", "gbm", "ggplot2", "dplyr"))

library(caret)
library(pROC)
library(randomForest)
library(rpart)
library(gbm)
library(ggplot2)
library(dplyr)


# ============================================================
# 2. Load Dataset
# ============================================================

# Option 1: Recommended GitHub structure
# Place the dataset in a folder named "data" and use this line:
# df <- read.csv("data/telco_customer_churn.csv", stringsAsFactors = FALSE)

# Option 2: Choose the dataset manually from your computer
df <- read.csv(file.choose(), stringsAsFactors = FALSE)

# Basic data checks
dim(df)
head(df)
names(df)
str(df)
summary(df)
colSums(is.na(df))


# ============================================================
# 3. Target Variable Check
# ============================================================

table(df$Churn)
prop.table(table(df$Churn))

# Convert Churn into a factor variable
# "Yes" is treated as the positive class because the goal is to detect churn.
df$Churn <- factor(df$Churn, levels = c("Yes", "No"))

table(df$Churn)


# ============================================================
# 4. Train-Test Split
# ============================================================

set.seed(42)

train_index <- createDataPartition(df$Churn, p = 0.8, list = FALSE)

train_data <- df[train_index, ]
test_data <- df[-train_index, ]

dim(train_data)
dim(test_data)

table(train_data$Churn)
table(test_data$Churn)


# ============================================================
# 5. One-Hot Encoding and Feature Preparation
# ============================================================

dummy_model <- dummyVars(
  Churn ~ .,
  data = train_data,
  fullRank = TRUE
)

x_train <- predict(dummy_model, newdata = train_data)
x_test <- predict(dummy_model, newdata = test_data)

x_train <- as.data.frame(x_train)
x_test <- as.data.frame(x_test)

y_train <- train_data$Churn
y_test <- test_data$Churn

# Remove near-zero variance predictors
nzv <- nearZeroVar(x_train)

if (length(nzv) > 0) {
  x_train <- x_train[, -nzv]
  x_test <- x_test[, -nzv]
}

dim(x_train)
dim(x_test)


# ============================================================
# 6. Custom Evaluation Function
# ============================================================

# F1-score is used as the main model selection metric because
# churn detection usually requires balancing Precision and Recall.
f1_summary <- function(data, lev = NULL, model = NULL) {

  cm <- confusionMatrix(
    data$pred,
    data$obs,
    positive = "Yes"
  )

  out <- c(
    F1 = as.numeric(cm$byClass["F1"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall = as.numeric(cm$byClass["Recall"]),
    Accuracy = as.numeric(cm$overall["Accuracy"])
  )

  return(out)
}

control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = f1_summary,
  sampling = "up",
  savePredictions = "final"
)


# ============================================================
# 7. Helper Function for Model Evaluation
# ============================================================

evaluate_model <- function(model, model_name, x_test, y_test) {

  pred_class <- predict(model, newdata = x_test)
  pred_prob <- predict(model, newdata = x_test, type = "prob")

  cm <- confusionMatrix(
    pred_class,
    y_test,
    positive = "Yes"
  )

  roc_obj <- roc(
    response = y_test,
    predictor = pred_prob[, "Yes"],
    levels = c("No", "Yes"),
    direction = "<"
  )

  result <- data.frame(
    Model = model_name,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall = as.numeric(cm$byClass["Recall"]),
    F1_score = as.numeric(cm$byClass["F1"]),
    ROC_AUC = as.numeric(auc(roc_obj))
  )

  return(list(
    confusion_matrix = cm,
    roc = roc_obj,
    result = result
  ))
}


# ============================================================
# 8. Logistic Regression Baseline
# ============================================================

set.seed(42)

model_logistic <- train(
  x = x_train,
  y = y_train,
  method = "glm",
  family = binomial(),
  trControl = control,
  metric = "F1"
)

logistic_eval <- evaluate_model(
  model = model_logistic,
  model_name = "Logistic Regression",
  x_test = x_test,
  y_test = y_test
)

logistic_eval$confusion_matrix
logistic_result <- logistic_eval$result
logistic_result


# ============================================================
# 9. Decision Tree
# ============================================================

set.seed(42)

model_tree <- train(
  x = x_train,
  y = y_train,
  method = "rpart",
  trControl = control,
  metric = "F1",
  tuneLength = 10
)

tree_eval <- evaluate_model(
  model = model_tree,
  model_name = "Decision Tree",
  x_test = x_test,
  y_test = y_test
)

tree_eval$confusion_matrix
tree_result <- tree_eval$result
tree_result

# Decision Tree tuning plot
tree_results <- model_tree$results
best_cp <- model_tree$bestTune$cp

best_tree_point <- tree_results %>%
  filter(cp == best_cp)

p_tree_tuning <- ggplot(tree_results, aes(x = cp, y = F1)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_point(
    data = best_tree_point,
    aes(x = cp, y = F1),
    size = 4
  ) +
  geom_vline(
    xintercept = best_cp,
    linetype = "dashed",
    linewidth = 1
  ) +
  annotate(
    "label",
    x = best_cp * 2.2,
    y = max(tree_results$F1, na.rm = TRUE) - 0.001,
    label = paste0("Best cp = ", round(best_cp, 4)),
    fontface = "bold",
    fill = "white",
    size = 4.5
  ) +
  scale_x_log10(
    breaks = c(0.003, 0.01, 0.03, 0.10),
    labels = c("0.003", "0.010", "0.030", "0.100")
  ) +
  scale_y_continuous(
    labels = function(x) sprintf("%.3f", x),
    expand = expansion(mult = c(0.05, 0.10))
  ) +
  labs(
    title = "Decision Tree Hyperparameter Tuning",
    subtitle = "Best complexity parameter selected based on cross-validated F1-score",
    x = "Complexity Parameter (cp, log scale)",
    y = "Cross-validated F1-score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

p_tree_tuning

# Decision Tree variable importance
tree_importance <- varImp(model_tree)

tree_imp_df <- data.frame(
  Variable = rownames(tree_importance$importance),
  Importance = tree_importance$importance$Overall
) %>%
  filter(Importance > 0) %>%
  arrange(desc(Importance)) %>%
  head(10)

p_tree_importance <- ggplot(tree_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  geom_text(aes(label = round(Importance, 1)), hjust = -0.1, size = 4) +
  coord_flip() +
  labs(
    title = "Top 10 Important Variables - Decision Tree",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold")
  )

p_tree_importance


# ============================================================
# 10. K-Nearest Neighbors
# ============================================================

set.seed(42)

model_knn <- train(
  x = x_train,
  y = y_train,
  method = "knn",
  trControl = control,
  metric = "F1",
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(k = seq(5, 101, by = 5))
)

knn_eval <- evaluate_model(
  model = model_knn,
  model_name = "KNN",
  x_test = x_test,
  y_test = y_test
)

knn_eval$confusion_matrix
knn_result <- knn_eval$result
knn_result

# KNN tuning plot
knn_results <- model_knn$results

p_knn_tuning <- ggplot(knn_results, aes(x = k, y = F1)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  geom_vline(
    xintercept = model_knn$bestTune$k,
    linetype = "dashed",
    linewidth = 1
  ) +
  annotate(
    "label",
    x = model_knn$bestTune$k + 10,
    y = max(knn_results$F1, na.rm = TRUE) - 0.004,
    label = paste("Best k =", model_knn$bestTune$k),
    fontface = "bold",
    size = 4.5,
    fill = "white"
  ) +
  labs(
    title = "KNN Hyperparameter Tuning",
    subtitle = "Cross-validated F1-score across different values of k",
    x = "Number of Neighbors (k)",
    y = "Cross-validated F1-score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title = element_text(face = "bold")
  )

p_knn_tuning

# KNN permutation importance based on F1-score
base_pred <- predict(model_knn, newdata = x_test)

base_cm <- confusionMatrix(
  base_pred,
  y_test,
  positive = "Yes"
)

base_f1 <- as.numeric(base_cm$byClass["F1"])

knn_importance_f1 <- data.frame()

for (var in names(x_test)) {

  x_perm <- x_test
  x_perm[[var]] <- sample(x_perm[[var]])

  perm_pred <- predict(model_knn, newdata = x_perm)

  perm_cm <- confusionMatrix(
    perm_pred,
    y_test,
    positive = "Yes"
  )

  perm_f1 <- as.numeric(perm_cm$byClass["F1"])
  importance_value <- base_f1 - perm_f1

  knn_importance_f1 <- rbind(
    knn_importance_f1,
    data.frame(
      Variable = var,
      Baseline_F1 = base_f1,
      Permuted_F1 = perm_f1,
      Importance = importance_value
    )
  )
}

top_knn_importance_f1 <- knn_importance_f1 %>%
  filter(Importance > 0) %>%
  slice_max(Importance, n = 10)

top_knn_importance_f1$Importance_Scaled <-
  top_knn_importance_f1$Importance / max(top_knn_importance_f1$Importance) * 100

p_knn_importance <- ggplot(
  top_knn_importance_f1,
  aes(x = reorder(Variable, Importance_Scaled), y = Importance_Scaled)
) +
  geom_col() +
  geom_text(aes(label = round(Importance_Scaled, 1)), hjust = -0.1, size = 4) +
  coord_flip() +
  ylim(0, 110) +
  labs(
    title = "Top 10 Important Variables - KNN",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold")
  )

p_knn_importance


# ============================================================
# 11. Random Forest
# ============================================================

set.seed(42)

rf_grid <- expand.grid(
  mtry = unique(pmax(
    1,
    round(seq(1, ncol(x_train), length.out = min(8, ncol(x_train))))
  ))
)

model_rf <- train(
  x = x_train,
  y = y_train,
  method = "rf",
  trControl = control,
  metric = "F1",
  tuneGrid = rf_grid,
  ntree = 500
)

rf_eval <- evaluate_model(
  model = model_rf,
  model_name = "Random Forest",
  x_test = x_test,
  y_test = y_test
)

rf_eval$confusion_matrix
rf_result <- rf_eval$result
rf_result

# Random Forest tuning plot
rf_results <- model_rf$results
best_mtry <- model_rf$bestTune$mtry

best_rf_point <- rf_results %>%
  filter(mtry == best_mtry)

p_rf_tuning <- ggplot(rf_results, aes(x = mtry, y = F1)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_point(
    data = best_rf_point,
    aes(x = mtry, y = F1),
    size = 4
  ) +
  geom_vline(
    xintercept = best_mtry,
    linetype = "dashed",
    linewidth = 1
  ) +
  annotate(
    "label",
    x = best_mtry + 1.2,
    y = max(rf_results$F1, na.rm = TRUE) - 0.001,
    label = paste0("Best mtry = ", best_mtry),
    fontface = "bold",
    fill = "white",
    size = 4.5
  ) +
  scale_y_continuous(
    labels = function(x) sprintf("%.3f", x),
    expand = expansion(mult = c(0.05, 0.10))
  ) +
  labs(
    title = "Random Forest Hyperparameter Tuning",
    subtitle = "Best mtry selected based on cross-validated F1-score",
    x = "Number of Variables Randomly Sampled at Each Split (mtry)",
    y = "Cross-validated F1-score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

p_rf_tuning

# Random Forest variable importance
rf_importance <- varImp(model_rf)

rf_imp_df <- data.frame(
  Variable = rownames(rf_importance$importance),
  Importance = rf_importance$importance$Overall
) %>%
  arrange(desc(Importance)) %>%
  head(10)

p_rf_importance <- ggplot(rf_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  geom_text(
    aes(label = round(Importance, 1)),
    hjust = -0.1,
    size = 4,
    fontface = "bold"
  ) +
  coord_flip() +
  ylim(0, max(rf_imp_df$Importance) * 1.15) +
  labs(
    title = "Top 10 Important Variables - Random Forest",
    subtitle = "Variables with highest contribution to churn prediction",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14)

p_rf_importance


# ============================================================
# 12. Gradient Boosting
# ============================================================

set.seed(42)

model_gbm <- train(
  x = x_train,
  y = y_train,
  method = "gbm",
  trControl = control,
  metric = "F1",
  verbose = FALSE,
  tuneGrid = expand.grid(
    n.trees = c(100, 200),
    interaction.depth = c(1, 3),
    shrinkage = c(0.05, 0.1),
    n.minobsinnode = c(10)
  )
)

gbm_eval <- evaluate_model(
  model = model_gbm,
  model_name = "Gradient Boosting",
  x_test = x_test,
  y_test = y_test
)

gbm_eval$confusion_matrix
gbm_result <- gbm_eval$result
gbm_result

# Gradient Boosting tuning heatmap
gbm_tuning_heatmap <- model_gbm$results %>%
  mutate(
    Trees = factor(n.trees),
    Learning_Rate = factor(shrinkage),
    Depth = paste("Tree Depth =", interaction.depth),
    Best_Model = ifelse(
      n.trees == model_gbm$bestTune$n.trees &
        interaction.depth == model_gbm$bestTune$interaction.depth &
        shrinkage == model_gbm$bestTune$shrinkage &
        n.minobsinnode == model_gbm$bestTune$n.minobsinnode,
      "Best",
      "Other"
    )
  )

p_gbm_tuning <- ggplot(
  gbm_tuning_heatmap,
  aes(x = Trees, y = Learning_Rate, fill = F1)
) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(
    aes(label = round(F1, 3)),
    fontface = "bold",
    size = 4.5
  ) +
  geom_tile(
    data = gbm_tuning_heatmap %>% filter(Best_Model == "Best"),
    aes(x = Trees, y = Learning_Rate),
    fill = NA,
    linewidth = 1.8
  ) +
  facet_wrap(~ Depth) +
  labs(
    title = "Gradient Boosting Hyperparameter Tuning",
    subtitle = "Cross-validated F1-score by number of trees, learning rate, and tree depth",
    x = "Number of Trees",
    y = "Learning Rate / Shrinkage",
    fill = "F1-score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 13)
  )

p_gbm_tuning

# Gradient Boosting variable importance
gbm_importance <- varImp(model_gbm)

gbm_imp_df <- data.frame(
  Variable = rownames(gbm_importance$importance),
  Importance = gbm_importance$importance$Overall
) %>%
  arrange(desc(Importance)) %>%
  head(10)

p_gbm_importance <- ggplot(gbm_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  geom_text(
    aes(label = round(Importance, 1)),
    hjust = -0.1,
    size = 4,
    fontface = "bold"
  ) +
  coord_flip() +
  ylim(0, max(gbm_imp_df$Importance) * 1.15) +
  labs(
    title = "Top 10 Important Variables - Gradient Boosting",
    subtitle = "Variables with highest contribution to churn prediction",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14)

p_gbm_importance


# ============================================================
# 13. Model Comparison
# ============================================================

dir.create("results", showWarnings = FALSE)

final_results <- bind_rows(
  logistic_result,
  tree_result,
  knn_result,
  rf_result,
  gbm_result
) %>%
  mutate(
    Accuracy = round(Accuracy, 4),
    Precision = round(Precision, 4),
    Recall = round(Recall, 4),
    F1_score = round(F1_score, 4),
    ROC_AUC = round(ROC_AUC, 4)
  ) %>%
  arrange(desc(F1_score), desc(ROC_AUC), desc(Recall))

final_results

write.csv(
  final_results,
  "results/model_comparison_results.csv",
  row.names = FALSE
)

best_model <- final_results[1, ]

cat("\nBest model based on F1-score:\n")
print(best_model)


# ============================================================
# 14. F1-Score Ranking Plot
# ============================================================

f1_plot_data <- final_results %>%
  arrange(F1_score) %>%
  mutate(
    Model = factor(Model, levels = Model),
    Best_Model = ifelse(
      F1_score == max(F1_score),
      "Best model",
      "Other models"
    )
  )

mean_f1 <- mean(f1_plot_data$F1_score, na.rm = TRUE)

x_min <- max(0, min(f1_plot_data$F1_score, na.rm = TRUE) - 0.02)
x_max <- min(1, max(f1_plot_data$F1_score, na.rm = TRUE) + 0.04)

p_f1_rank <- ggplot(
  f1_plot_data,
  aes(x = F1_score, y = Model)
) +
  geom_vline(
    xintercept = mean_f1,
    linetype = "dashed",
    linewidth = 0.9
  ) +
  geom_segment(
    aes(
      x = x_min,
      xend = F1_score,
      y = Model,
      yend = Model
    ),
    linewidth = 1.1
  ) +
  geom_point(
    aes(shape = Best_Model),
    size = 5
  ) +
  geom_text(
    aes(label = sprintf("%.3f", F1_score)),
    hjust = -0.35,
    size = 4.8,
    fontface = "bold"
  ) +
  coord_cartesian(
    xlim = c(x_min, x_max),
    clip = "off"
  ) +
  labs(
    title = "F1-Score Ranking of Machine Learning Models",
    subtitle = paste0(
      "F1-score is the main criterion for model selection | Mean F1 = ",
      sprintf("%.3f", mean_f1)
    ),
    x = "F1-score on Test Set",
    y = "Model",
    shape = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold", size = 12),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.margin = ggplot2::margin(10, 45, 10, 10)
  )

p_f1_rank

ggsave(
  filename = "results/model_f1_ranking.png",
  plot = p_f1_rank,
  width = 10,
  height = 6,
  dpi = 300
)


# ============================================================
# 15. Random Forest Threshold Optimization
# ============================================================

rf_cv_pred <- model_rf$pred

rf_cv_pred_best <- rf_cv_pred %>%
  filter(mtry == model_rf$bestTune$mtry)

thresholds <- seq(0.1, 0.9, by = 0.01)

threshold_results <- data.frame()

for (t in thresholds) {

  pred_class <- ifelse(rf_cv_pred_best$Yes >= t, "Yes", "No")
  pred_class <- factor(pred_class, levels = levels(rf_cv_pred_best$obs))

  cm <- confusionMatrix(
    pred_class,
    rf_cv_pred_best$obs,
    positive = "Yes"
  )

  temp <- data.frame(
    Threshold = t,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall = as.numeric(cm$byClass["Recall"]),
    F1_score = as.numeric(cm$byClass["F1"])
  )

  threshold_results <- rbind(threshold_results, temp)
}

threshold_results[is.na(threshold_results)] <- 0

best_threshold <- threshold_results$Threshold[
  which.max(threshold_results$F1_score)
]

best_threshold

threshold_results %>%
  arrange(desc(F1_score)) %>%
  head(10)

p_threshold <- ggplot(threshold_results, aes(x = Threshold, y = F1_score)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  geom_vline(
    xintercept = best_threshold,
    linetype = "dashed",
    linewidth = 1
  ) +
  annotate(
    "label",
    x = best_threshold,
    y = max(threshold_results$F1_score, na.rm = TRUE),
    label = paste0("Best threshold = ", round(best_threshold, 2)),
    fontface = "bold",
    fill = "white"
  ) +
  labs(
    title = "Random Forest Threshold Optimization",
    subtitle = "Threshold selected based on cross-validated F1-score",
    x = "Classification Threshold",
    y = "Cross-validated F1-score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(face = "bold")
  )

p_threshold

ggsave(
  filename = "results/random_forest_threshold_optimization.png",
  plot = p_threshold,
  width = 10,
  height = 6,
  dpi = 300
)


# ============================================================
# 16. Evaluate Random Forest with Optimized Threshold on Test Set
# ============================================================

prob_rf_test <- predict(model_rf, newdata = x_test, type = "prob")[, "Yes"]

pred_rf_threshold <- ifelse(prob_rf_test >= best_threshold, "Yes", "No")
pred_rf_threshold <- factor(pred_rf_threshold, levels = levels(y_test))

cm_rf_threshold <- confusionMatrix(
  pred_rf_threshold,
  y_test,
  positive = "Yes"
)

cm_rf_threshold

rf_threshold_result <- data.frame(
  Model = "Random Forest - Optimized Threshold",
  Threshold = best_threshold,
  Accuracy = as.numeric(cm_rf_threshold$overall["Accuracy"]),
  Precision = as.numeric(cm_rf_threshold$byClass["Precision"]),
  Recall = as.numeric(cm_rf_threshold$byClass["Recall"]),
  F1_score = as.numeric(cm_rf_threshold$byClass["F1"]),
  ROC_AUC = as.numeric(rf_result$ROC_AUC)
)

rf_threshold_result <- rf_threshold_result %>%
  mutate(
    Threshold = round(Threshold, 2),
    Accuracy = round(Accuracy, 4),
    Precision = round(Precision, 4),
    Recall = round(Recall, 4),
    F1_score = round(F1_score, 4),
    ROC_AUC = round(ROC_AUC, 4)
  )

rf_threshold_result

write.csv(
  rf_threshold_result,
  "results/random_forest_threshold_result.csv",
  row.names = FALSE
)

cat("\nScript completed successfully. Results are saved in the 'results' folder.\n")
