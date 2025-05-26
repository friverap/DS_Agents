import dspy
import agents.memory_agents as m
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import logging
from utils.logger import Logger
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = Logger("agents", see_time=True, console_log=False)

# Configure DSPy with a default language model
try:
    # Use the ModelManager for DSPy configuration
    from utils.model_utils import model_manager
    logger.log_message("DSPy configured using ModelManager", level=logging.INFO)
except Exception as e:
    logger.log_message(f"DSPy configuration error: {str(e)}", level=logging.WARNING)
    # Fallback configuration attempt
    try:
        import dspy
        dspy.configure()
        logger.log_message("DSPy fallback configuration successful", level=logging.INFO)
    except Exception as fallback_error:
        logger.log_message(f"Fallback DSPy configuration also failed: {str(fallback_error)}", level=logging.ERROR)

AGENTS_WITH_DESCRIPTION = {
    "preprocessing_agent": "Cleans and prepares a DataFrame using Pandas and NumPy—handles missing values, detects column types, and converts date strings to datetime.",
    "statistical_analytics_agent": "Performs statistical analysis (e.g., regression, seasonal decomposition) using statsmodels, with proper handling of categorical data and missing values.",
    "sk_learn_agent": "Trains and evaluates machine learning models using scikit-learn, including classification, regression, and clustering with feature importance insights.",
    "data_viz_agent": "Generates interactive visualizations with Plotly, selecting the best chart type to reveal trends, comparisons, and insights based on the analysis goal.",
    "web_search_agent": "Searches the web using Brave Search API to find current information, research topics, and gather external data sources."
}

PLANNER_AGENTS_WITH_DESCRIPTION = {
    "planner_preprocessing_agent": (
        "Cleans and prepares a DataFrame using Pandas and NumPy, "
        "handles missing values, detects column types, and converts date strings to datetime. "
        "Outputs a cleaned DataFrame for the planner_statistical_analytics_agent."
    ),
    "planner_statistical_analytics_agent": (
        "Takes the cleaned DataFrame from preprocessing, performs statistical analysis "
        "(e.g., regression, seasonal decomposition) using statsmodels with proper handling "
        "of categorical data and remaining missing values. "
        "Produces summary statistics and model diagnostics for the planner_sk_learn_agent."
    ),
    "planner_sk_learn_agent": (
        "Receives summary statistics and the cleaned data, trains and evaluates machine "
        "learning models using scikit-learn (classification, regression, clustering), "
        "and generates performance metrics and feature importance. "
        "Passes the trained models and evaluation results to the planner_data_viz_agent."
    ),
    "planner_data_viz_agent": (
        "Consumes trained models and evaluation results to create interactive visualizations "
        "with Plotly—selects the best chart type, applies styling, and annotates insights. "
        "Delivers ready-to-share figures that communicate model performance and key findings."
    ),
    "planner_web_search_agent": (
        "Searches the web for relevant information, current trends, and external data sources "
        "to enhance analysis context and provide up-to-date insights."
    )
}

def get_agent_description(agent_name, is_planner=False):
    if is_planner:
        return PLANNER_AGENTS_WITH_DESCRIPTION[agent_name.lower()] if agent_name.lower() in PLANNER_AGENTS_WITH_DESCRIPTION else "No description available for this agent"
    else:
        return AGENTS_WITH_DESCRIPTION[agent_name.lower()] if agent_name.lower() in AGENTS_WITH_DESCRIPTION else "No description available for this agent"


# Agent to make a Chat history name from a query
class chat_history_name_agent(dspy.Signature):
    """You are an agent that takes a query and returns a name for the chat history"""
    query = dspy.InputField(desc="The query to make a name for")
    name = dspy.OutputField(desc="A name for the chat history (max 3 words)")

class dataset_description_agent(dspy.Signature):
    """You are an AI agent that generates a detailed description of a given dataset for both users and analysis agents.
Your description should serve two key purposes:
1. Provide users with context about the dataset's purpose, structure, and key attributes.
2. Give analysis agents critical data handling instructions to prevent common errors.
For data handling instructions, you must always include Python data types and address the following:
- Data type warnings (e.g., numeric columns stored as strings that need conversion).
- Null value handling recommendations.
- Format inconsistencies that require preprocessing.
- Explicit warnings about columns that appear numeric but are stored as strings (e.g., '10' vs 10).
- Explicit Python data types for each major column (e.g., int, float, str, bool, datetime).
- Columns with numeric values that should be treated as categorical (e.g., zip codes, IDs).
- Any date parsing or standardization required (e.g., MM/DD/YYYY to datetime).
- Any other technical considerations that would affect downstream analysis or modeling.
- List all columns and their data types with exact case sensitive spelling
If an existing description is provided, enhance it with both business context and technical guidance for analysis agents, preserving accurate information from the existing description or what the user has written.
Ensure the description is comprehensive and provides actionable insights for both users and analysis agents.
Example:
This housing dataset contains property details including price, square footage, bedrooms, and location data.
It provides insights into real estate market trends across different neighborhoods and property types.
TECHNICAL CONSIDERATIONS FOR ANALYSIS:
- price (str): Appears numeric but is stored as strings with a '$' prefix and commas (e.g., "$350,000"). Requires cleaning with str.replace('$','').replace(',','') and conversion to float.
- square_footage (str): Contains unit suffix like 'sq ft' (e.g., "1,200 sq ft"). Remove suffix and commas before converting to int.
- bedrooms (int): Correctly typed but may contain null values (~5% missing) – consider imputation or filtering.
- zip_code (int): Numeric column but should be treated as str or category to preserve leading zeros and prevent unintended numerical analysis.
- year_built (float): May contain missing values (~15%) – consider mean/median imputation or exclusion depending on use case.
- listing_date (str): Dates stored in "MM/DD/YYYY" format – convert to datetime using pd.to_datetime().
- property_type (str): Categorical column with inconsistent capitalization (e.g., "Condo", "condo", "CONDO") – normalize to lowercase for consistent grouping.
    """
    dataset = dspy.InputField(desc="The dataset to describe, including headers, sample data, null counts, and data types.")
    existing_description = dspy.InputField(desc="An existing description to improve upon (if provided).", default="")
    description = dspy.OutputField(desc="A comprehensive dataset description with business context and technical guidance for analysis agents.")


class advanced_query_planner(dspy.Signature):
    """
You are a advanced data analytics planner agent. Your task is to generate the most efficient plan—using the fewest necessary agents and variables—to achieve a user-defined goal. The plan must preserve data integrity, avoid unnecessary steps, and ensure clear data flow between agents.
Inputs:
1. Datasets (raw or preprocessed)
2. Agent descriptions (roles, variables they create/use, constraints)
3. User-defined goal (e.g., prediction, analysis, visualization)
Responsibilities:
1. Feasibility: Confirm the goal is achievable with the provided data and agents; ask for clarification if it's unclear.
2. Minimal Plan: Use the smallest set of agents and variables; avoid redundant transformations; ensure agents are ordered logically and only included if essential.
3. Instructions: For each agent, define:
   * create: output variables and their purpose
   * use: input variables and their role
   * instruction: concise explanation of the agent's function and relevance to the goal
4. Clarity: Keep instructions precise; avoid intermediate steps unless necessary; ensure each agent has a distinct, relevant role.
### Output Format:
Example: 1 agent use
  goal: "Generate a bar plot showing sales by category after cleaning the raw data and calculating the average of the 'sales' column"
Output:
  plan: planner_data_viz_agent
{
  "planner_data_viz_agent": {
    "create": [
      "cleaned_data: DataFrame - cleaned version of df (pd.Dataframe) after removing null values"
    ],
    "use": [
      "df: DataFrame - unprocessed dataset (pd.Dataframe) containing sales and category information"
    ],
    "instruction": "Clean df by removing null values, calculate the average sales, and generate a bar plot showing sales by category."
  }
}
Example 3 Agent 
goal:"Clean the dataset, run a linear regression to model the relationship between marketing budget and sales, and visualize the regression line with confidence intervals."
plan: planner_preprocessing_agent -> planner_statistical_analytics_agent -> planner_data_viz_agent
{
  "planner_preprocessing_agent": {
    "create": [
      "cleaned_data: DataFrame - cleaned version of df with missing values handled and proper data types inferred"
    ],
    "use": [
      "df: DataFrame - dataset containing marketing budgets and sales figures"
    ],
    "instruction": "Clean df by handling missing values and converting column types (e.g., dates). Output cleaned_data for modeling."
  },
  "planner_statistical_analytics_agent": {
    "create": [
      "regression_results: dict - model summary including coefficients, p-values, R², and confidence intervals"
    ],
    "use": [
      "cleaned_data: DataFrame - preprocessed dataset ready for regression"
    ],
    "instruction": "Perform linear regression using cleaned_data to model sales as a function of marketing budget. Return regression_results including coefficients and confidence intervals."
  },
  "planner_data_viz_agent": {
    "create": [
      "regression_plot: PlotlyFigure - visual plot showing regression line with confidence intervals"
    ],
    "use": [
      "cleaned_data: DataFrame - original dataset for plotting",
      "regression_results: dict - output of linear regression"
    ],
    "instruction": "Generate a Plotly regression plot using cleaned_data and regression_results. Show the fitted line, scatter points, and 95% confidence intervals."
  }
}
Try to use as few agents to answer the user query as possible.
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df, columns set df as copy of df")
    Agent_desc = dspy.InputField(desc="The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal")

    plan = dspy.OutputField(desc="The plan that would achieve the user defined goal", prefix='Plan:')
    plan_instructions = dspy.OutputField(desc="Detailed variable-level instructions per agent for the plan")

class basic_query_planner(dspy.Signature):
    """
    You are the basic query planner in the system, you pick one agent, to answer the user's goal.
    Use the Agent_desc that describes the names and actions of agents available.
    Example: Visualize height and salary?
    plan:planner_data_viz_agent
    plan_instructions:
               {
                    "planner_data_viz_agent": {
                        "create": ["scatter_plot"],
                        "use": ["original_data"],
                        "instruction": "use the original_data to create scatter_plot of height & salary, using plotly"
                    }
                }
    Example: Tell me the correlation between X and Y
    plan:planner_preprocessing_agent
    plan_instructions:{
                    "planner_data_viz_agent": {
                        "create": ["correlation"],
                        "use": ["original_data"],
                        "instruction": "use the original_data to measure correlation of X & Y, using pandas"
                    }
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df, columns set df as copy of df")
    Agent_desc = dspy.InputField(desc="The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal")
    plan = dspy.OutputField(desc="The plan that would achieve the user defined goal", prefix='Plan:')
    plan_instructions = dspy.OutputField(desc="Instructions on what the agent should do alone")



class intermediate_query_planner(dspy.Signature):
    # The planner agent which routes the query to Agent(s)
    # The output is like this Agent1->Agent2 etc
    """ You are an intermediate data analytics planner agent. You have access to three inputs
    1. Data available in the system
    2. Data Agent descriptions
    3. User-defined Goal
    You take these three inputs to develop a comprehensive plan to achieve the user-defined goal from the data & Agents available.
    In case you think the user-defined goal is infeasible you can ask the user to redefine or add more description to the goal.
    Give your output in this format:
    plan: Agent1->Agent2
    plan_instructions = {
    "Agent1": {
                        "create": ["aggregated_variable"],
                        "use": ["original_data"],
                        "instruction": "use the original_data to create aggregated_variable"
                    },
    "Agent2": {
                        "create": ["visualization_of_data"],
                        "use": ["aggregated_variable,original_data"],
                        "instruction": "use the aggregated_variable & original_data to create visualization_of_data"
                    }
            }
    Keep the instructions minimal without many variables, and minimize the number of unknowns, keep it obvious!
    Try to use no more than 2 agents, unless completely necessary!
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df,columns  set df as copy of df")
    Agent_desc = dspy.InputField(desc= "The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal ")
    plan = dspy.OutputField(desc="The plan that would achieve the user defined goal", prefix='Plan:')
    plan_instructions= dspy.OutputField(desc="Instructions from the planner")

class goal_refiner_agent(dspy.Signature):
    """
    You are a goal refinement agent that takes user queries and refines them for better analysis.
    Your task is to clarify ambiguous requests and suggest specific, actionable goals.
    """
    original_goal = dspy.InputField(desc="The original user goal or query")
    context = dspy.InputField(desc="Additional context about the data or domain", default="")
    
    refined_goal = dspy.OutputField(desc="A refined, specific, and actionable goal")
    suggestions = dspy.OutputField(desc="Additional suggestions for analysis")

class planner_module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.basic_planner = dspy.ChainOfThought(basic_query_planner)
        self.intermediate_planner = dspy.ChainOfThought(intermediate_query_planner)
        self.advanced_planner = dspy.ChainOfThought(advanced_query_planner)
        self.goal_refiner = dspy.ChainOfThought(goal_refiner_agent)
        
    def forward(self, goal, dataset, Agent_desc):
        # Determine complexity based on goal
        goal_lower = goal.lower()
        
        # Simple visualization or basic analysis
        if any(keyword in goal_lower for keyword in ['plot', 'chart', 'visualize', 'graph', 'show']):
            if not any(keyword in goal_lower for keyword in ['clean', 'preprocess', 'model', 'predict', 'regression', 'classification']):
                logger.log_message("Using basic planner for simple visualization", level=logging.INFO)
                return self.basic_planner(goal=goal, dataset=dataset, Agent_desc=Agent_desc)
        
        # Complex analysis requiring multiple steps
        if any(keyword in goal_lower for keyword in ['model', 'predict', 'regression', 'classification', 'cluster', 'machine learning', 'ml']):
            logger.log_message("Using advanced planner for complex analysis", level=logging.INFO)
            return self.advanced_planner(goal=goal, dataset=dataset, Agent_desc=Agent_desc)
        
        # Default to intermediate planner
        logger.log_message("Using intermediate planner for moderate complexity", level=logging.INFO)
        return self.intermediate_planner(goal=goal, dataset=dataset, Agent_desc=Agent_desc)

class planner_preprocessing_agent(dspy.Signature):
    """
You are a preprocessing agent in a multi-agent data analytics system.
You are given:
* A  dataset  (already loaded as `df`).
* A  user-defined analysis goal  (e.g., predictive modeling, exploration, cleaning).
*  Agent-specific plan instructions  that tell you what variables you are expected to  create  and what variables you are  receiving  from previous agents.
* processed_df is just an arbitrary name, it can be anything the planner says to clean!
### Your Responsibilities:
*  Follow the provided plan and create only the required variables listed in the 'create' section of the plan instructions.
*  Do not create fake data  or introduce variables not explicitly part of the instructions.
*  Do not read data from CSV ; the dataset (`df`) is already loaded and ready for processing.
* Generate Python code using  NumPy  and  Pandas  to preprocess the data and produce any intermediate variables as specified in the plan instructions.
### Best Practices for Preprocessing:
1.  Create a copy of the original DataFrame : It will always be stored as df, it already exists use it!
    ```python
    processed_df = df.copy()
    ```
2.  Separate column types :
    ```python
    numeric_cols = processed_df.select_dtypes(include='number').columns
    categorical_cols = processed_df.select_dtypes(include='object').columns
    ```
3.  Handle missing values :
    ```python
    for col in numeric_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    for col in categorical_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'Unknown')
    ```
4.  Convert string columns to datetime safely :
    ```python
    def safe_to_datetime(x):
        try:
            return pd.to_datetime(x, errors='coerce', cache=False)
        except (ValueError, TypeError):
            return pd.NaT
    
    cleaned_df['date_column'] = cleaned_df['date_column'].apply(safe_to_datetime)
    ```
> Replace `processed_df`,'cleaned_df' and `date_column` with whatever names the user or planner provides.
5.  Do not alter the DataFrame index :
   Avoid using `reset_index()`, `set_index()`, or reindexing unless explicitly instructed.
6.  Log assumptions and corrections  in comments to clarify any choices made during preprocessing.
7.  Do not mutate global state : Avoid in-place modifications unless clearly necessary (e.g., using `.copy()`).
8.  Handle data types properly :
   * Avoid coercing types blindly (e.g., don't compare timestamps to strings or floats).
   * Use `pd.to_datetime(..., errors='coerce')` for safe datetime parsing.
9.  Preserve column structure : Only drop or rename columns if explicitly instructed.
### Output:
1.  Code : Python code that performs the requested preprocessing steps as per the plan instructions.
2.  Summary : A brief explanation of what preprocessing was done (e.g., columns handled, missing value treatment).
### Principles to Follow:
-Never alter the DataFrame index  unless explicitly instructed.
-Handle missing data  explicitly, filling with default values when necessary.
-Preserve column structure  and avoid unnecessary modifications.
-Ensure data types are appropriate  (e.g., dates parsed correctly).
-Log assumptions  in the code.
    """
    dataset = dspy.InputField(desc="The dataset, preloaded as df")
    goal = dspy.InputField(desc="User-defined goal for the analysis")
    plan_instructions = dspy.InputField(desc="Agent-level instructions about what to create and receive", format=str)
    
    code = dspy.OutputField(desc="Generated Python code for preprocessing")
    summary = dspy.OutputField(desc="Explanation of what was done and why")

class planner_data_viz_agent(dspy.Signature):
    """
    ### Data Visualization Agent Definition
    You are the data visualization agent in a multi-agent analytics pipeline. Your primary responsibility is to generate visualizations based on the user-defined goal and the plan instructions.
    You are provided with:
    * goal: A user-defined goal outlining the type of visualization the user wants (e.g., "plot sales over time with trendline").
    * dataset: The dataset (e.g., `df_cleaned`) which will be passed to you by other agents in the pipeline. Do not assume or create any variables — the data is already present and valid when you receive it.
    * styling_index: Specific styling instructions (e.g., axis formatting, color schemes) for the visualization.
    * plan_instructions: A dictionary containing:
    * 'create': List of visualization components you must generate (e.g., 'scatter_plot', 'bar_chart').
    * 'use': List of variables you must use to generate the visualizations. This includes datasets and any other variables provided by the other agents.
    * 'instructions': A list of additional instructions related to the creation of the visualizations, such as requests for trendlines or axis formats.
    ---
    ### Responsibilities:
    1. Strict Use of Provided Variables:
    * You must never create fake data. Only use the variables and datasets that are explicitly provided to you in the `plan_instructions['use']` section. All the required data must already be available.
    * If any variable listed in `plan_instructions['use']` is missing or invalid, you must return an error and not proceed with any visualization.
    2. Visualization Creation:
    * Based on the 'create' section of the `plan_instructions`, generate the required visualization using Plotly. For example, if the goal is to plot a time series, you might generate a line chart.
    * Respect the user-defined goal in determining which type of visualization to create.
    3. Performance Optimization:
    * If the dataset contains more than 50,000 rows, you must sample the data to 5,000 rows to improve performance. Use this method:
        ```python
        if len(df) > 50000:
            df = df.sample(5000, random_state=42)
        ```
    4. Layout and Styling:
    * Apply formatting and layout adjustments as defined by the styling_index. This may include:
        * Axis labels and title formatting.
        * Tick formats for axes.
        * Color schemes or color maps for visual elements.
    * You must ensure that all axes (x and y) have consistent formats (e.g., using `K`, `M`, or 1,000 format, but not mixing formats).
    5. Trendlines:
    * Trendlines should only be included if explicitly requested in the 'instructions' section of `plan_instructions`.
    6. Displaying the Visualization:
    * Use Plotly's `fig.show()` method to display the created chart.
    * Never output raw datasets or the goal itself. Only the visualization code and the chart should be returned.
    7. Error Handling:
    * If the required dataset or variables are missing or invalid (i.e., not included in `plan_instructions['use']`), return an error message indicating which specific variable is missing or invalid.
    * If the goal or create instructions are ambiguous or invalid, return an error stating the issue.
    8. No Data Modification:
    * Never modify the provided dataset or generate new data. If the data needs preprocessing or cleaning, assume it's already been done by other agents.
    ---
    ### Strict Conditions:
    * You never create any data.
    * You only use the data and variables passed to you.
    * If any required data or variable is missing or invalid, you must stop and return a clear error message.
    * it should be update_yaxes, update_xaxes, not axis
    By following these conditions and responsibilities, your role is to ensure that the visualizations are generated as per the user goal, using the valid data and instructions given to you.
        """
    goal = dspy.InputField(desc="User-defined chart goal (e.g. trendlines, scatter plots)")
    dataset = dspy.InputField(desc="Details of the dataframe (`df`) and its columns")
    styling_index = dspy.InputField(desc="Instructions for plot styling and layout formatting")
    plan_instructions = dspy.InputField(desc="Variables to create and receive for visualization purposes", format=str)

    code = dspy.OutputField(desc="Plotly Python code for the visualization")
    summary = dspy.OutputField(desc="Plain-language summary of what is being visualized")

class planner_statistical_analytics_agent(dspy.Signature):
    """
Agent Definition:
You are a statistical analytics agent in a multi-agent data analytics pipeline.
You are given:
* A dataset (usually a cleaned or transformed version like `df_cleaned`).
* A user-defined goal (e.g., regression, seasonal decomposition).
* Agent-specific plan instructions specifying:
  * Which variables you are expected to CREATE (e.g., `regression_model`).
  * Which variables you will USE (e.g., `df_cleaned`, `target_variable`).
  * A set of instructions outlining additional processing or handling for these variables (e.g., handling missing values, adding constants, transforming features, etc.).
Your Responsibilities:
* Use the `statsmodels` library to implement the required statistical analysis.
* Ensure that all strings are handled as categorical variables via `C(col)` in model formulas.
* Always add a constant using `sm.add_constant()`.
* Do not modify the DataFrame's index.
* Convert `X` and `y` to float before fitting the model.
* Handle missing values before modeling.
* Avoid any data visualization (that is handled by another agent).
* Write output to the console using `print()`.
If the goal is regression:
* Use `statsmodels.OLS` with proper handling of categorical variables and adding a constant term.
* Handle missing values appropriately.
If the goal is seasonal decomposition:
* Use `statsmodels.tsa.seasonal_decompose`.
* Ensure the time series and period are correctly provided (i.e., `period` should not be `None`).
You must not:
* You must always create the variables in `plan_instructions['CREATE']`.
* Never create the `df` variable. Only work with the variables passed via the `plan_instructions`.
* Rely on hardcoded column names — use those passed via `plan_instructions`.
* Introduce or modify intermediate variables unless they are explicitly listed in `plan_instructions['CREATE']`.
Instructions to Follow:
1. CREATE only the variables specified in `plan_instructions['CREATE']`. Do not create any intermediate or new variables.
2. USE only the variables specified in `plan_instructions['USE']` to carry out the task.
3. Follow any additional instructions in `plan_instructions['INSTRUCTIONS']` (e.g., preprocessing steps, encoding, handling missing values).
4. Do not reassign or modify any variables passed via `plan_instructions`. These should be used as-is.
Example Workflow:
Given that the `plan_instructions` specifies variables to CREATE and USE, and includes instructions, your approach should look like this:
1. Use `df_cleaned` and the variables like `X` and `y` from `plan_instructions` for analysis.
2. Follow instructions for preprocessing (e.g., handle missing values or scale features).
3. If the goal is regression:
   * Use `sm.OLS` for model fitting.
   * Handle categorical variables via `C(col)` and add a constant term.
4. If the goal is seasonal decomposition:
   * Ensure `period` is provided and use `sm.tsa.seasonal_decompose`.
5. Store the output variable as specified in `plan_instructions['CREATE']`.
### Example Code Structure:
```python
import statsmodels.api as sm
def statistical_model(X, y, goal, period=None):
    try:
        # Check for missing values and handle them
        X = X.dropna()
        y = y.loc[X.index].dropna()
        # Ensure X and y are aligned
        X = X.loc[y.index]
        # Convert categorical variables
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category')
        # Add a constant term to the predictor
        X = sm.add_constant(X)
        # Fit the model
        if goal == 'regression':
            # Handle categorical variables in the model formula
            formula = 'y ~ ' + ' + '.join([f'C({col})' if X[col].dtype.name == 'category' else col for col in X.columns])
            model = sm.OLS(y.astype(float), X.astype(float)).fit()
            return model.summary()
        elif goal == 'seasonal_decompose':
            if period is None:
                raise ValueError("Period must be specified for seasonal decomposition")
            decomposition = sm.tsa.seasonal_decompose(y, period=period)
            return decomposition
        else:
            raise ValueError("Unknown goal specified.")
        
    except Exception as e:
        return f"An error occurred: {e}"
```
Summary:
1. Always USE the variables passed in `plan_instructions['USE']` to carry out the task.
2. Only CREATE the variables specified in `plan_instructions['CREATE']`. Do not create any additional variables.
3. Follow any additional instructions in `plan_instructions['INSTRUCTIONS']` (e.g., handling missing values, adding constants).
4. Ensure reproducibility by setting the random state appropriately and handling categorical variables.
5. Focus on statistical analysis and avoid any unnecessary data manipulation.
Output:
* The code implementing the statistical analysis, including all required steps.
* A summary of what the statistical analysis does, how it's performed, and why it fits the goal.
    """
    dataset = dspy.InputField(desc="Preprocessed dataset, often named df_cleaned")
    goal = dspy.InputField(desc="The user's statistical analysis goal, e.g., regression or seasonal_decompose")
    plan_instructions = dspy.InputField(desc="Instructions on variables to create and receive for statistical modeling", format=str)
    
    code = dspy.OutputField(desc="Python code for statistical modeling using statsmodels")
    summary = dspy.OutputField(desc="Explanation of statistical analysis steps")

class planner_sk_learn_agent(dspy.Signature):
    """
    Agent Definition:
    You are a machine learning agent in a multi-agent data analytics pipeline.
    You are given:
    * A dataset (often cleaned and feature-engineered).
    * A user-defined goal (e.g., classification, regression, clustering).
    * Agent-specific plan instructions specifying:
    * Which variables you are expected to CREATE (e.g., `trained_model`, `predictions`).
    * Which variables you will USE (e.g., `df_cleaned`, `target_variable`, `feature_columns`).
    * A set of instructions outlining additional processing or handling for these variables (e.g., handling missing values, applying transformations, or other task-specific guidelines).
    Your Responsibilities:
    * Use the scikit-learn library to implement the appropriate ML pipeline.
    * Always split data into training and testing sets where applicable.
    * Use `print()` for all outputs.
    * Ensure your code is:
    * Reproducible: Set `random_state=42` wherever applicable.
    * Modular: Avoid deeply nested code.
    * Focused on model building, not visualization (leave plotting to the `data_viz_agent`).
    * Your task may include:
    * Preprocessing inputs (e.g., encoding).
    * Model selection and training.
    * Evaluation (e.g., accuracy, RMSE, classification report).
    You must not:
    * Visualize anything (that's another agent's job).
    * Rely on hardcoded column names — use those passed via `plan_instructions`.
    * Never create or modify any variables not explicitly mentioned in `plan_instructions['CREATE']`.
    * Never create the `df` variable. You will only work with the variables passed via the `plan_instructions`.
    * Do not introduce intermediate variables unless they are listed in `plan_instructions['CREATE']`.
    Instructions to Follow:
    1. CREATE only the variables specified in the `plan_instructions['CREATE']` list. Do not create any intermediate or new variables.
    2. USE only the variables specified in the `plan_instructions['USE']` list. You are not allowed to create or modify any variables not listed in the plan instructions.
    3. Follow any processing instructions in the `plan_instructions['INSTRUCTIONS']` list. This might include tasks like handling missing values, scaling features, or encoding categorical variables. Always perform these steps on the variables specified in the `plan_instructions`.
    4. Do not reassign or modify any variables passed via `plan_instructions`. These should be used as-is.
    Example Workflow:
    Given that the `plan_instructions` specifies variables to CREATE and USE, and includes instructions, your approach should look like this:
    1. Use `df_cleaned` and `feature_columns` from the `plan_instructions` to extract your features (`X`).
    2. Use `target_column` from `plan_instructions` to extract your target (`y`).
    3. If instructions are provided (e.g., scale or encode), follow them.
    4. Split data into training and testing sets using `train_test_split`.
    5. Train the model based on the received goal (classification, regression, etc.).
    6. Store the output variables as specified in `plan_instructions['CREATE']`.
    ### Example Code Structure:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    # Ensure that all variables follow plan instructions:
    # Use received inputs: df_cleaned, feature_columns, target_column
    X = df_cleaned[feature_columns]
    y = df_cleaned[target_column]
    # Apply any preprocessing instructions (e.g., scaling if instructed)
    if 'scale' in plan_instructions['INSTRUCTIONS']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Select and train the model (based on the task)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    # Generate predictions
    predictions = model.predict(X_test)
    # Create the variable specified in 'plan_instructions': 'metrics'
    metrics = classification_report(y_test, predictions)
    # Print the results
    print(metrics)
    # Ensure the 'metrics' variable is returned as requested in the plan
    ```
    Summary:
    1. Always USE the variables passed in `plan_instructions['USE']` to build the pipeline.
    2. Only CREATE the variables specified in `plan_instructions['CREATE']`. Do not create any additional variables.
    3. Follow any additional instructions in `plan_instructions['INSTRUCTIONS']` (e.g., preprocessing steps).
    4. Ensure reproducibility by setting `random_state=42` wherever necessary.
    5. Focus on model building, evaluation, and saving the required outputs—avoid any unnecessary variables.
    Output:
    * The code implementing the ML task, including all required steps.
    * A summary of what the model does, how it is evaluated, and why it fits the goal.
    """
    dataset = dspy.InputField(desc="Input dataset, often cleaned and feature-selected (e.g., df_cleaned)")
    goal = dspy.InputField(desc="The user's machine learning goal (e.g., classification or regression)")
    plan_instructions = dspy.InputField(desc="Instructions indicating what to create and what variables to receive", format=str)

    code = dspy.OutputField(desc="Scikit-learn based machine learning code")
    summary = dspy.OutputField(desc="Explanation of the ML approach and evaluation")

class story_teller_agent(dspy.Signature):
    # Optional helper agent, which can be called to build a analytics story 
    # For all of the analysis performed
    """ You are a story teller agent, taking output from different data analytics agents, you compose a compelling story for what was done """
    agent_analysis_list = dspy.InputField(desc="A list of analysis descriptions from every agent", format=str)
    story = dspy.OutputField(desc="A coherent story combining the whole analysis")

class code_combiner_agent(dspy.Signature):
    # Combines code from different agents into one script
    """ You are a code combine agent, taking Python code output from many agents and combining the operations into 1 output
    You also fix any errors in the code. 
    IMPORTANT: You may be provided with previous interaction history. The section marked "### Current Query:" contains the user's current request. Any text in "### Previous Interaction History:" is for context only and is NOT part of the current request.
    Double check column_names/dtypes using dataset, also check if applied logic works for the datatype
    df = df.copy()
    Also add this to display Plotly chart
    fig.show()
    Make sure your output is as intended!
    Provide a concise bullet-point summary of the code integration performed.
    
    Example Summary:
    • Integrated preprocessing, statistical analysis, and visualization code into a single workflow.
    • Fixed variable scope issues, standardized DataFrame handling (e.g., using `df.copy()`), and corrected errors.
    • Validated column names and data types against the dataset definition to prevent runtime issues.
    • Ensured visualizations are displayed correctly (e.g., added `fig.show()`).
    """
    dataset = dspy.InputField(desc="Use this double check column_names, data types")
    agent_code_list = dspy.InputField(desc="A list of code given by each agent", format=str)
    refined_complete_code = dspy.OutputField(desc="Refined complete code base")
    summary = dspy.OutputField(desc="A concise 4 bullet-point summary of the code integration performed and improvements made")

class data_viz_agent(dspy.Signature):
    # Visualizes data using Plotly
    """    
    You are an AI agent responsible for generating interactive data visualizations using Plotly.
    IMPORTANT Instructions:
    - The section marked "### Current Query:" contains the user's request. Any text in "### Previous Interaction History:" is for context only and should NOT be treated as part of the current request.
    - You must only use the tools provided to you. This agent handles visualization only.
    - If len(df) > 50000, always sample the dataset before visualization using:  
    if len(df) > 50000:  
        df = df.sample(50000, random_state=1)
    - Each visualization must be generated as a separate figure using go.Figure().  
    Do NOT use subplots under any circumstances.
    - Each figure must be returned individually using:  
    fig.to_html(full_html=False)
    - Use update_layout with xaxis and yaxis only once per figure.
    - Enhance readability and clarity by:  
    • Using low opacity (0.4-0.7) where appropriate  
    • Applying visually distinct colors for different elements or categories  
    - Make sure the visual answers the user's specific goal:  
    • Identify what insight or comparison the user is trying to achieve  
    • Choose the visualization type and features (e.g., color, size, grouping) to emphasize that goal  
    • For example, if the user asks for "trends in revenue," use a time series line chart; if they ask for "top-performing categories," use a bar chart sorted by value  
    • Prioritize highlighting patterns, outliers, or comparisons relevant to the question
    - Never include the dataset or styling index in the output.
    - If there are no relevant columns for the requested visualization, respond with:  
    "No relevant columns found to generate this visualization."
    - Use only one number format consistently: either 'K', 'M', or comma-separated values like 1,000/1,000,000. Do not mix formats.
    - Only include trendlines in scatter plots if the user explicitly asks for them.
    - Output only the code and a concise bullet-point summary of what the visualization reveals.
    - Always end each visualization with:  
    fig.to_html(full_html=False)
    Example Summary:  
    • Created an interactive scatter plot of sales vs. marketing spend with color-coded product categories  
    • Included a trend line showing positive correlation (r=0.72)  
    • Highlighted outliers where high marketing spend resulted in low sales  
    • Generated a time series chart of monthly revenue from 2020-2023  
    • Added annotations for key business events  
    • Visualization reveals 35% YoY growth with seasonal peaks in Q4
    """
    goal = dspy.InputField(desc="user defined goal which includes information about data and chart they want to plot")
    dataset = dspy.InputField(desc=" Provides information about the data in the data frame. Only use column names and dataframe_name as in this context")
    styling_index = dspy.InputField(desc='Provides instructions on how to style your Plotly plots')
    code= dspy.OutputField(desc="Plotly code that visualizes what the user needs according to the query & dataframe_index & styling_context")
    summary = dspy.OutputField(desc="A concise bullet-point summary of the visualization created and key insights revealed")

class code_fix(dspy.Signature):
    """
You are an expert AI developer and data analyst assistant, skilled at identifying and resolving issues in Python code related to data analytics. Another agent has attempted to generate Python code for a data analytics task but produced code that is broken or throws an error.
Your task is to:
1. Carefully examine the provided faulty_code and the corresponding error message.
2. Identify the exact cause of the failure based on the error and surrounding context.
3. Modify only the necessary portion(s) of the code to fix the issue, utilizing the dataset_context to inform your corrections.
4. Ensure the intended behavior of the original code is preserved (e.g., if the code is meant to filter, group, or visualize data, that functionality must be preserved).
5. Ensure the final output is runnable, error-free, and logically consistent.
Strict instructions:
- Assume the dataset is already loaded and available in the code context; do not include any code to read, load, or create data.
- Do not modify any working parts of the code unnecessarily.
- Do not change variable names, structure, or logic unless it directly contributes to resolving the issue.
- Do not output anything besides the corrected, full version of the code (i.e., no explanations, comments, or logs).
- Avoid introducing new dependencies or libraries unless absolutely required to fix the problem.
- The output must be complete and executable as-is.
Be precise, minimal, and reliable. Prioritize functional correctness.
One-shot example:
===
dataset_context: 
"This dataset contains historical price and trading data for two major financial assets: the S&P 500 index and Bitcoin (BTC). The data includes daily price metrics (open, high, low, close) and percentage changes for both assets... Change % columns are stored as strings with '%' symbol (e.g., '-5.97%') and require cleaning."
faulty_code:
# Convert percentage strings to floats
df['Change %'] = df['Change %'].str.rstrip('%').astype(float)
df['Change % BTC'] = df['Change % BTC'].str.rstrip('%').astype(float)
error:
Error in data_viz_agent: Can only use .str accessor with string values!
Traceback (most recent call last):
  File "/app/scripts/format_response.py", line 196, in execute_code_from_markdown
    exec(block_code, context)
AttributeError: Can only use .str accessor with string values!
fixed_code:
# Convert percentage strings to floats
df['Change %'] = df['Change %'].astype(str).str.rstrip('%').astype(float)
df['Change % BTC'] = df['Change % BTC'].astype(str).str.rstrip('%').astype(float)
===
    """
    dataset_context = dspy.InputField(desc="The dataset context to be used for the code fix")
    faulty_code = dspy.InputField(desc="The faulty Python code used for data analytics")
    error = dspy.InputField(desc="The error message thrown when running the code")
    fixed_code = dspy.OutputField(desc="The corrected and executable version of the code")

class code_edit(dspy.Signature):
    """
You are an expert AI code editor that specializes in modifying existing data analytics code based on user requests. The user provides a working or partially working code snippet, a natural language prompt describing the desired change, and dataset context information.
Your job is to:
1. Analyze the provided original_code, user_prompt, and dataset_context.
2. Modify only the part(s) of the code that are relevant to the user's request, using the dataset context to inform your edits.
3. Leave all unrelated parts of the code unchanged, unless the user explicitly requests a full rewrite or broader changes.
4. Ensure that your changes maintain or improve the functionality and correctness of the code.
Strict requirements:
- Assume the dataset is already loaded and available in the code context; do not include any code to read, load, or create data.
- Do not change variable names, function structures, or logic outside the scope of the user's request.
- Do not refactor, optimize, or rewrite unless explicitly instructed.
- Ensure the edited code remains complete and executable.
- Output only the modified code, without any additional explanation, comments, or extra formatting.
Make your edits precise, minimal, and faithful to the user's instructions, using the dataset context to guide your modifications.
    """
    dataset_context = dspy.InputField(desc="The dataset context to be used for the code edit, including information about the dataset's shape, columns, types, and null values")
    original_code = dspy.InputField(desc="The original code the user wants modified")
    user_prompt = dspy.InputField(desc="The user instruction describing how the code should be changed")
    edited_code = dspy.OutputField(desc="The updated version of the code reflecting the user's request, incorporating changes informed by the dataset context")

class preprocessing_agent(dspy.Signature):
    """
    You are a data preprocessing agent. Your task is to clean and prepare datasets for analysis.
    """
    dataset = dspy.InputField(desc="The dataset to preprocess")
    goal = dspy.InputField(desc="The preprocessing goal")
    
    code = dspy.OutputField(desc="Python code for preprocessing")
    summary = dspy.OutputField(desc="Summary of preprocessing steps")

class statistical_analytics_agent(dspy.Signature):
    """
    You are a statistical analysis agent. Your task is to perform statistical analysis on datasets.
    """
    dataset = dspy.InputField(desc="The dataset to analyze")
    goal = dspy.InputField(desc="The statistical analysis goal")
    
    code = dspy.OutputField(desc="Python code for statistical analysis")
    summary = dspy.OutputField(desc="Summary of statistical analysis")

class sk_learn_agent(dspy.Signature):
    """
    You are a machine learning agent using scikit-learn. Your task is to build and evaluate ML models.
    """
    dataset = dspy.InputField(desc="The dataset for machine learning")
    goal = dspy.InputField(desc="The machine learning goal")
    
    code = dspy.OutputField(desc="Python code for machine learning")
    summary = dspy.OutputField(desc="Summary of machine learning approach")

class auto_analyst_ind(dspy.Module):
    # Individual agent execution module
    def __init__(self, agents, retrievers):
        # Initialize agent modules and retrievers
        super().__init__()
        self.agents = agents
        self.retrievers = retrievers
        
        # Initialize all agent modules
        self.preprocessing_agent = dspy.ChainOfThought(preprocessing_agent)
        self.statistical_analytics_agent = dspy.ChainOfThought(statistical_analytics_agent)
        self.sk_learn_agent = dspy.ChainOfThought(sk_learn_agent)
        self.data_viz_agent = dspy.ChainOfThought(data_viz_agent)
        self.story_teller_agent = dspy.ChainOfThought(story_teller_agent)
        self.code_combiner_agent = dspy.ChainOfThought(code_combiner_agent)
        self.code_fix = dspy.ChainOfThought(code_fix)
        self.code_edit = dspy.ChainOfThought(code_edit)
        
        # Initialize memory agents
        self.memory_agent = m.memory_agent()
        self.memory_summarize_agent = m.memory_summarize_agent()
        
    def execute_agent(self, specified_agent, inputs):
        # Execute a specific agent with given inputs
        agent_module = getattr(self, specified_agent, None)
        if agent_module:
            return agent_module(**inputs)
        else:
            raise ValueError(f"Agent {specified_agent} not found")
    
    def execute_agent_with_memory(self, specified_agent, inputs, query):
        # Execute agent with memory context
        try:
            # Get memory context
            memory_context = self.memory_agent.forward(query=query)
            
            # Add memory context to inputs
            if 'dataset' in inputs:
                inputs['dataset'] = f"{inputs['dataset']}\n\nMemory Context: {memory_context.memory_context}"
            
            # Execute the agent
            result = self.execute_agent(specified_agent, inputs)
            
            # Update memory with result
            self.memory_agent.update_memory(query, str(result))
            
            return result
        except Exception as e:
            logger.log_message(f"Error executing agent with memory: {str(e)}", level=logging.ERROR)
            # Fallback to regular execution
            return self.execute_agent(specified_agent, inputs)
    
    def forward(self, query, specified_agent):
        # Main forward method for individual agent execution
        try:
            # Prepare inputs based on agent type
            if specified_agent in ['preprocessing_agent', 'statistical_analytics_agent', 'sk_learn_agent']:
                inputs = {
                    'dataset': self.retrievers["dataframe_index"].as_retriever().retrieve(query)[0].text,
                    'goal': query
                }
            elif specified_agent == 'data_viz_agent':
                inputs = {
                    'goal': query,
                    'dataset': self.retrievers["dataframe_index"].as_retriever().retrieve(query)[0].text,
                    'styling_index': self.retrievers["style_index"].as_retriever().retrieve(query)[0].text
                }
            elif specified_agent == 'story_teller_agent':
                inputs = {
                    'agent_analysis_list': query  # Assuming query contains analysis list
                }
            elif specified_agent == 'code_combiner_agent':
                inputs = {
                    'dataset': self.retrievers["dataframe_index"].as_retriever().retrieve(query)[0].text,
                    'agent_code_list': query  # Assuming query contains code list
                }
            elif specified_agent in ['code_fix', 'code_edit']:
                # These agents need special input handling
                inputs = query  # Assuming query is already a dict with required fields
            else:
                inputs = {'query': query}
            
            # Execute with memory if available
            return self.execute_agent_with_memory(specified_agent, inputs, query)
            
        except Exception as e:
            logger.log_message(f"Error in auto_analyst_ind forward: {str(e)}", level=logging.ERROR)
            return {"error": str(e)}
    
    def execute_multiple_agents(self, query, agent_list):
        # Execute multiple agents in sequence
        results = {}
        accumulated_context = ""
        
        for agent_name in agent_list:
            try:
                # Add accumulated context to query for subsequent agents
                enhanced_query = f"{query}\n\nPrevious Results: {accumulated_context}" if accumulated_context else query
                
                result = self.forward(enhanced_query, agent_name)
                results[agent_name] = result
                
                # Accumulate context for next agent
                if isinstance(result, dict):
                    if 'summary' in result:
                        accumulated_context += f"\n{agent_name}: {result['summary']}"
                    elif 'code' in result:
                        accumulated_context += f"\n{agent_name}: Generated code"
                else:
                    accumulated_context += f"\n{agent_name}: {str(result)[:200]}..."
                    
            except Exception as e:
                logger.log_message(f"Error executing agent {agent_name}: {str(e)}", level=logging.ERROR)
                results[agent_name] = {"error": str(e)}
        
        return results

class auto_analyst(dspy.Module):
    # Main auto analyst module with planning capabilities
    def __init__(self, agents, retrievers):
        # Initialize agent modules and retrievers
        super().__init__()
        self.agents = agents
        self.retrievers = retrievers
        
        # Initialize planner modules
        self.planner_module = planner_module()
        self.planner_preprocessing_agent = dspy.ChainOfThought(planner_preprocessing_agent)
        self.planner_statistical_analytics_agent = dspy.ChainOfThought(planner_statistical_analytics_agent)
        self.planner_sk_learn_agent = dspy.ChainOfThought(planner_sk_learn_agent)
        self.planner_data_viz_agent = dspy.ChainOfThought(planner_data_viz_agent)
        
        # Initialize individual agent modules
        self.preprocessing_agent = dspy.ChainOfThought(preprocessing_agent)
        self.statistical_analytics_agent = dspy.ChainOfThought(statistical_analytics_agent)
        self.sk_learn_agent = dspy.ChainOfThought(sk_learn_agent)
        self.data_viz_agent = dspy.ChainOfThought(data_viz_agent)
        self.story_teller_agent = dspy.ChainOfThought(story_teller_agent)
        self.code_combiner_agent = dspy.ChainOfThought(code_combiner_agent)
        self.code_fix = dspy.ChainOfThought(code_fix)
        self.code_edit = dspy.ChainOfThought(code_edit)
        self.goal_refiner_agent = dspy.ChainOfThought(goal_refiner_agent)
        
        # Initialize memory agents
        self.memory_agent = m.memory_agent()
        self.memory_summarize_agent = m.memory_summarize_agent()
        
        # Initialize dataset description agent
        self.dataset_description_agent = dspy.ChainOfThought(dataset_description_agent)
        
        # Initialize chat history name agent
        self.chat_history_name_agent = dspy.ChainOfThought(chat_history_name_agent)
    
    def execute_agent(self, agent_name, inputs):
        # Execute a specific agent with given inputs
        agent_module = getattr(self, agent_name, None)
        if agent_module:
            return agent_module(**inputs)
        else:
            raise ValueError(f"Agent {agent_name} not found")
    
    def get_plan(self, query):
        # Get execution plan for the query using the planner module
        try:
            # Prepare dataset description
            dataset_desc = "df - DataFrame with uploaded data containing user's dataset"
            
            # Prepare agent descriptions
            agent_desc = """Available agents:
            - planner_preprocessing_agent: Cleans and prepares data, handles missing values, converts data types
            - planner_statistical_analytics_agent: Performs statistical analysis, regression, correlation analysis
            - planner_sk_learn_agent: Builds machine learning models, classification, regression, clustering
            - planner_data_viz_agent: Creates interactive visualizations with Plotly
            - planner_web_search_agent: Searches web for relevant information and data sources"""
            
            # Use the planner module to generate a plan
            logger.log_message(f"Using planner module to generate plan for query: {query}", level=logging.INFO)
            plan_result = self.planner_module.forward(
                goal=query,
                dataset=dataset_desc,
                Agent_desc=agent_desc
            )
            
            logger.log_message(f"Planner module result: {plan_result}", level=logging.INFO)
            
            # Extract plan and instructions from the result
            if hasattr(plan_result, 'plan') and hasattr(plan_result, 'plan_instructions'):
                # Parse plan_instructions if it's a string
                plan_instructions = plan_result.plan_instructions
                if isinstance(plan_instructions, str):
                    try:
                        # Try to parse as JSON if it's a string
                        import json
                        plan_instructions = json.loads(plan_instructions)
                    except:
                        # If parsing fails, create a simple structure
                        plan_instructions = {
                            plan_result.plan: {
                                "create": ["analysis_results"],
                                "use": ["df"],
                                "instruction": plan_instructions
                            }
                        }
                
                # Create a plan object
                class DetailedPlan:
                    def __init__(self, plan_text, instructions):
                        self.plan = plan_text
                        self.plan_instructions = instructions
                
                return DetailedPlan(plan_result.plan, plan_instructions)
            
            else:
                # Fallback to simple rule-based planning
                logger.log_message("Planner module didn't return expected format, using fallback", level=logging.WARNING)
                return self._get_simple_plan(query)
            
        except Exception as e:
            logger.log_message(f"Error using planner module: {str(e)}, falling back to simple planning", level=logging.WARNING)
            return self._get_simple_plan(query)
    
    def _get_simple_plan(self, query):
        # Fallback simple rule-based planning
        try:
            query_lower = query.lower()
            
            # Determine which agents to use based on keywords
            if any(keyword in query_lower for keyword in ['visualize', 'plot', 'chart', 'graph', 'show']):
                plan = "planner_data_viz_agent"
                plan_instructions = {
                    "planner_data_viz_agent": {
                        "create": ["visualization"],
                        "use": ["df"],
                        "instruction": "Create a visualization based on the user's request using the uploaded dataset"
                    }
                }
            elif any(keyword in query_lower for keyword in ['eda', 'exploratory', 'analyze', 'analysis', 'statistics', 'statistical', 'correlation']):
                plan = "planner_statistical_analytics_agent"
                plan_instructions = {
                    "planner_statistical_analytics_agent": {
                        "create": ["statistical_analysis", "correlation_matrix", "summary_statistics"],
                        "use": ["df"],
                        "instruction": "Perform comprehensive statistical analysis including descriptive statistics, correlations, and data exploration"
                    }
                }
            elif any(keyword in query_lower for keyword in ['clean', 'preprocess', 'prepare', 'missing values']):
                plan = "planner_preprocessing_agent"
                plan_instructions = {
                    "planner_preprocessing_agent": {
                        "create": ["cleaned_data", "preprocessing_report"],
                        "use": ["df"],
                        "instruction": "Clean and preprocess the dataset, handle missing values, and prepare data for analysis"
                    }
                }
            elif any(keyword in query_lower for keyword in ['model', 'predict', 'machine learning', 'ml', 'classification', 'regression']):
                plan = "planner_sk_learn_agent"
                plan_instructions = {
                    "planner_sk_learn_agent": {
                        "create": ["trained_model", "model_metrics", "predictions"],
                        "use": ["df"],
                        "instruction": "Build and evaluate machine learning models for prediction or classification"
                    }
                }
            else:
                # Default to statistical analysis for EDA-like requests
                plan = "planner_statistical_analytics_agent"
                plan_instructions = {
                    "planner_statistical_analytics_agent": {
                        "create": ["exploratory_analysis", "data_insights"],
                        "use": ["df"],
                        "instruction": "Perform exploratory data analysis to understand the dataset structure and patterns"
                    }
                }
            
            # Create a simple plan object
            class SimplePlan:
                def __init__(self, plan_text, instructions):
                    self.plan = plan_text
                    self.plan_instructions = instructions
            
            return SimplePlan(plan, plan_instructions)
            
        except Exception as e:
            logger.log_message(f"Error in simple planning: {str(e)}", level=logging.ERROR)
            return {"error": str(e)}
    
    def execute_plan(self, query, plan):
        # Execute the planned sequence of agents with real data
        try:
            # Parse plan and plan_instructions
            plan_text = plan.plan if hasattr(plan, 'plan') else str(plan)
            plan_instructions = plan.plan_instructions if hasattr(plan, 'plan_instructions') else {}
            
            # Handle different types of plan_instructions
            if isinstance(plan_instructions, str):
                try:
                    # Try to parse as JSON if it's a string
                    import json
                    plan_instructions = json.loads(plan_instructions)
                except (json.JSONDecodeError, ValueError):
                    # If parsing fails, create a simple structure
                    plan_instructions = {"instruction": plan_instructions}
            elif plan_instructions is None:
                plan_instructions = {}
            
            # Extract agent sequence from plan
            if '->' in plan_text:
                agent_sequence = [agent.strip() for agent in plan_text.split('->')]
            else:
                agent_sequence = [plan_text.strip()]
            
            results = {}
            
            # Execute agents in sequence with real implementations
            for agent_name in agent_sequence:
                try:
                    logger.log_message(f"Executing agent: {agent_name}", level=logging.INFO)
                    
                    # Get agent instructions for this specific agent
                    agent_instructions = plan_instructions.get(agent_name, {})
                    
                    # Ensure agent_instructions is a dict
                    if isinstance(agent_instructions, str):
                        agent_instructions = {"instruction": agent_instructions}
                    elif not isinstance(agent_instructions, dict):
                        agent_instructions = {"instruction": str(agent_instructions)}
                    
                    # Convert agent_instructions to JSON string for DSPy
                    import json
                    agent_instructions_str = json.dumps(agent_instructions, indent=2)
                    
                    # Prepare inputs for the agent
                    agent_inputs = {
                        "goal": query,
                        "dataset": "df",  # Assume df is available in context
                        "plan_instructions": agent_instructions_str
                    }
                    
                    # Execute the specific agent based on its type
                    if agent_name == 'planner_data_viz_agent':
                        # Execute data visualization agent
                        viz_agent = dspy.Predict(planner_data_viz_agent)
                        result = viz_agent(
                            goal=query,
                            dataset="df - DataFrame with uploaded data",
                            styling_index="Default styling with clear labels and colors",
                            plan_instructions=agent_instructions_str
                        )
                        results[agent_name] = {
                            "code": result.code,
                            "summary": result.summary,
                            "type": "visualization"
                        }
                        
                    elif agent_name == 'planner_statistical_analytics_agent':
                        # Execute statistical analysis agent
                        stats_agent = dspy.Predict(planner_statistical_analytics_agent)
                        result = stats_agent(
                            dataset="df - DataFrame with uploaded data",
                            goal=query,
                            plan_instructions=agent_instructions_str
                        )
                        results[agent_name] = {
                            "code": result.code,
                            "summary": result.summary,
                            "type": "statistical_analysis"
                        }
                        
                    elif agent_name == 'planner_preprocessing_agent':
                        # Execute preprocessing agent
                        prep_agent = dspy.Predict(planner_preprocessing_agent)
                        result = prep_agent(
                            dataset="df - DataFrame with uploaded data",
                            goal=query,
                            plan_instructions=agent_instructions_str
                        )
                        results[agent_name] = {
                            "code": result.code,
                            "summary": result.summary,
                            "type": "preprocessing"
                        }
                        
                    elif agent_name == 'planner_sk_learn_agent':
                        # Execute machine learning agent
                        ml_agent = dspy.Predict(planner_sk_learn_agent)
                        result = ml_agent(
                            dataset="df - DataFrame with uploaded data",
                            goal=query,
                            plan_instructions=agent_instructions_str
                        )
                        results[agent_name] = {
                            "code": result.code,
                            "summary": result.summary,
                            "type": "machine_learning"
                        }
                        
                    else:
                        # For any other agent, try to execute it generically
                        logger.log_message(f"Unknown agent type: {agent_name}, attempting generic execution", level=logging.WARNING)
                        results[agent_name] = {
                            "summary": f"Agent {agent_name} executed with generic handler",
                            "code": f"# {agent_name} execution\nprint('Agent {agent_name} completed')",
                            "type": "generic"
                        }
                    
                    logger.log_message(f"Successfully executed agent: {agent_name}", level=logging.INFO)
                    
                except Exception as e:
                    logger.log_message(f"Error executing agent {agent_name}: {str(e)}", level=logging.ERROR)
                    results[agent_name] = {
                        "error": str(e),
                        "type": "error"
                    }
            
            return results
            
        except Exception as e:
            logger.log_message(f"Error executing plan: {str(e)}", level=logging.ERROR)
            return {"error": str(e)}
    
    def execute_workflow(self, user_query: str, available_data: str = "") -> Dict[str, Any]:
        """
        Execute the appropriate workflow based on routing decision
        """
        try:
            logger.log_message(f"Starting execute_workflow with query: {user_query[:100]}...", level=logging.INFO)
            
            # First route the query to determine the approach
            routing_decision = self.route_query(user_query, available_data)
            logger.log_message(f"Routing decision: {routing_decision}", level=logging.INFO)
            
            if routing_decision.get("routing_decision") == "multi_agent":
                # Execute multi-agent workflow
                logger.log_message("Executing multi-agent workflow", level=logging.INFO)
                
                # Get plan for the query
                plan = self.get_plan(user_query)
                logger.log_message(f"Generated plan: {plan}", level=logging.INFO)
                
                if isinstance(plan, dict) and "error" in plan:
                    logger.log_message(f"Plan generation error: {plan['error']}", level=logging.ERROR)
                    return {
                        "type": "error",
                        "message": f"Planning error: {plan['error']}"
                    }
                
                # Format and display the plan first
                response_parts = []
                
                # Add plan summary
                if hasattr(plan, 'plan'):
                    response_parts.append(f"## 📋 Plan de Análisis\n**Agente seleccionado:** {plan.plan}")
                    
                    # Add plan instructions if available
                    if hasattr(plan, 'plan_instructions') and plan.plan_instructions:
                        response_parts.append(f"\n### 📝 Instrucciones del Plan:")
                        for agent_name, instructions in plan.plan_instructions.items():
                            agent_title = agent_name.replace('_', ' ').replace('planner ', '').title()
                            response_parts.append(f"\n**{agent_title}:**")
                            
                            if isinstance(instructions, dict):
                                if 'create' in instructions:
                                    response_parts.append(f"- **Crear:** {', '.join(instructions['create'])}")
                                if 'use' in instructions:
                                    response_parts.append(f"- **Usar:** {', '.join(instructions['use'])}")
                                if 'instruction' in instructions:
                                    response_parts.append(f"- **Instrucción:** {instructions['instruction']}")
                            else:
                                response_parts.append(f"- {str(instructions)}")
                    
                    response_parts.append(f"\n---\n\n## 🚀 Ejecutando Plan...")
                    logger.log_message(f"Added plan summary: {plan.plan}", level=logging.INFO)
                
                # Execute the plan
                logger.log_message("Executing plan with agents...", level=logging.INFO)
                results = self.execute_plan(user_query, plan)
                logger.log_message(f"Plan execution results: {results}", level=logging.INFO)
                
                if isinstance(results, dict) and "error" in results:
                    logger.log_message(f"Plan execution error: {results['error']}", level=logging.ERROR)
                    response_parts.append(f"\n### ❌ Error en la Ejecución:\n{results['error']}")
                    return {
                        "type": "error",
                        "response": "\n".join(response_parts),
                        "message": f"Execution error: {results['error']}"
                    }
                
                # Add agent execution results
                response_parts.append(f"\n## 🔍 Resultados de la Ejecución\n")
                
                for agent_name, result in results.items():
                    logger.log_message(f"Processing result for agent {agent_name}: {result}", level=logging.INFO)
                    agent_title = agent_name.replace('_', ' ').replace('planner ', '').title()
                    
                    if isinstance(result, dict):
                        if "error" in result:
                            response_parts.append(f"### ❌ Error en {agent_title}:\n{result['error']}\n")
                        else:
                            response_parts.append(f"### ✅ {agent_title}")
                            
                            if "summary" in result:
                                response_parts.append(f"\n**Resumen:** {result['summary']}\n")
                            
                            if "code" in result and result["code"]:
                                response_parts.append(f"\n**Código Generado:**\n```python\n{result['code']}\n```\n")
                    else:
                        response_parts.append(f"### {agent_title}\n{str(result)}\n")
                
                # Add execution summary
                successful_agents = len([r for r in results.values() if isinstance(r, dict) and "error" not in r])
                total_agents = len(results)
                response_parts.append(f"\n---\n\n## 📊 Resumen de Ejecución\n")
                response_parts.append(f"- **Agentes ejecutados exitosamente:** {successful_agents}/{total_agents}")
                response_parts.append(f"- **Plan completado:** {'✅ Sí' if successful_agents == total_agents else '⚠️ Con errores'}")
                
                final_response = "\n".join(response_parts)
                logger.log_message(f"Final formatted response length: {len(final_response)}", level=logging.INFO)
                
                return {
                    "type": "analysis",
                    "response": final_response,
                    "plan": plan,
                    "results": results,
                    "integration_summary": f"Executed {successful_agents}/{total_agents} agents successfully"
                }
            
            else:
                # Direct AI response
                logger.log_message("Using direct AI response", level=logging.INFO)
                return {
                    "type": "conversational",
                    "response": "This query will be handled by direct AI response."
                }
                
        except Exception as e:
            logger.log_message(f"Error in execute_workflow: {str(e)}", level=logging.ERROR)
            import traceback
            logger.log_message(f"Traceback: {traceback.format_exc()}", level=logging.ERROR)
            return {
                "type": "error",
                "message": f"Workflow execution error: {str(e)}"
            }
    
    def route_query(self, query: str, file_context: str = "") -> Dict[str, Any]:
        """
        Route query to determine if it should use multi-agent system or direct AI
        """
        try:
            query_lower = query.lower()
            
            # Keywords that indicate data analysis requests
            analysis_keywords = [
                'analyze', 'analysis', 'data', 'dataset', 'csv', 'plot', 'chart', 'graph',
                'visualize', 'visualization', 'statistics', 'statistical', 'correlation',
                'regression', 'model', 'predict', 'prediction', 'machine learning', 'ml',
                'classification', 'clustering', 'preprocessing', 'clean', 'explore',
                'exploratory', 'eda', 'distribution', 'trend', 'pattern'
            ]
            
            # Check if query contains analysis keywords
            is_analysis_request = any(keyword in query_lower for keyword in analysis_keywords)
            
            # Check if there's file context (uploaded files)
            has_file_context = bool(file_context and file_context.strip())
            
            # Determine routing
            if is_analysis_request or has_file_context:
                routing_decision = "multi_agent"
                agent_type = "analytical"
            else:
                routing_decision = "direct_ai"
                agent_type = "conversational"
            
            return {
                "routing_decision": routing_decision,
                "agent_type": agent_type,
                "is_analysis_request": is_analysis_request,
                "has_file_context": has_file_context,
                "confidence": 0.8 if is_analysis_request else 0.6
            }
            
        except Exception as e:
            logger.log_message(f"Error in route_query: {str(e)}", level=logging.ERROR)
            return {
                "routing_decision": "direct_ai",
                "agent_type": "conversational",
                "is_analysis_request": False,
                "has_file_context": False,
                "confidence": 0.5,
                "error": str(e)
            }
    
    def forward(self, query):
        # Main forward method
        try:
            # Get plan for the query
            plan = self.get_plan(query)
            
            if isinstance(plan, dict) and "error" in plan:
                return plan
            
            # Execute the plan
            results = self.execute_plan(query, plan)
            
            return {
                "plan": plan,
                "results": results,
                "query": query
            }
            
        except Exception as e:
            logger.log_message(f"Error in auto_analyst forward: {str(e)}", level=logging.ERROR)
            return {"error": str(e)}


# Global instance
multi_agent_system = None

def get_multi_agent_system():
    """Get or create the global multi-agent system instance."""
    global multi_agent_system
    if multi_agent_system is None:
        # Initialize with empty agents and retrievers
        agents = {}
        retrievers = {"dataframe_index": None, "style_index": None}
        multi_agent_system = auto_analyst(agents, retrievers)
    return multi_agent_system


def auto_analyst_ind(agents, retrievers):
    """Individual agent execution function"""
    return auto_analyst(agents, retrievers) 