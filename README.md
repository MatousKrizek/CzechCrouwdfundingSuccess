# CzechCrouwdfundingSuccess
This repository includes data and scripts used in the creation of my Bachelor thesis 'Determinants of Crowdfunding Success.' Data from crowdfunding websites Hithit.com and Startovac.cz is obtained, cleaned, and the data from Hithit.com is used for data analysis. The goal was to find out the shape of the influence of the goal (the money intended to obtain) of a crowdfunding campaign on the success probability dependent by category. This can help practicioners determine the size of goal they are likely to success with considering their type of project.

Included files & Story of the project:

1. Hithit data extraction branch
  1.1 HithitScraper.py - Iterates through project pages of crowdfunding projects on Hithit.com, downloading all their HTMLs. This one is simple, as there were no anti-bot measures on the site at the time of use.
  1.2 HitHit.pickle - The downloaded content of all the project pages except for those which are empty.
  1.3 HitHitFilter.py - Filters through the project pages, creating a Pandas dataframe HitHitDF used for the analysis.
   

2. Startovac data extraction branch
  2.1   StartovacLinkScraper.py - As in Startovac, the web addresses of the project pages are not named systematically, this script searches for all symbols of the      keayboard on the search bar and downloads all the contetn.
  2.2 LinksStartovacRaw.pickle - Content of all search results obtained by scipt (2.1). This file cannot be uploaded due to size (101 MB). - NOT INCLUDED
  2.3 StartovacSearchFilter.py - Filtering script to gather all the links from the search results (2.1). Result is a list of links to all found projects, purging       duplicates. The total number of links optained is close to the total number of all projects ever published on the platform, as I obtained this number from the platform owners. This way is also effective.
  2.4 LinksStartovac.pickle - List of links obtained from the downloaded content (2.2)
  2.5 StartovacScraper.py - Takes the list of links (2.4) and obtains the HTML of each project page. However, after roughly 1 000 pages, it is not able to capture the comment section, which is outsourced to a Facebook plugin, which is able to reveal this automathic scraping and refuses to give more information. As this is the only place I found on the Startovač website to give date of the project, I ended up with most of the project pages not including the date of beginning or ending, and therefore had to drop the observations from the data analysis, as the date or at least the year of the project was necessary for several variables.

3. Data analysis branch
   3.1 HitHitDF - Dataframe of all the projects captured on Hithit, which includes all projects published on the platform until march 2024, except for those which have been        later removed by HitHit on behalf of their authors.
   3.2 inflace.csv - CSV including data on inflation in the Czech Republic in the years of operation of the crowdfunding platforms.
   3.3 Crowdfunding Data Analysis - Analysis file created for Spyder 3.12. Includes cells:
   3.3.1 Technical - imports and proprietary function definitions
   3.3.2 Data Collection - imports the dataframe (3.1) and re-arrenges it selectively into new dataframe called cf, orchestrated for the needs of the analysis. Also imports data on inflation.
   3.3.3 Initial Data Cleaning - calculates and logically adds some new columns to the "cf" dataframe, changes types, handles NAs.
   3.3.4 Exploratory Data Analysis (EDA)
   3.3.4.1 Exploration through statistics - covariance, correlation and variance inflation factor are studied to test the usability of statistical methods.
   3.3.4.2 Graphical exploration
     - Histograms and box plots of distributions of all variables are generated
     - Scatter plot of distribution of projects in time (also by category)
     - Goal of individual categories (box plot)
     - Histogram of projects in time and their success rate and Point plot of success ratio in time (incl. version subtracting the Covid category)
     - Histogram of projects in time with highlighted surplus due to Covid 19 pandemic
     - Line plot and heatmap of the ratio of succesful projects in years per each category
     - Line plot of monthly ratio of succesful projects by year
  3.3.5 Post-EDA Adjustments - outliers that are further than 3 standard deviations from mean are removed, and new cleaned version of the dataframe is created.
  3.3.6 Pre-modeling check
       - Exports describing statistics of the treated data
       - Shows covariance and correlations among variables, if the use conditions of the models still stay true
       - Histograms and box plots of treated distributions of all numberic variables
       - Bar plots of the number of project in each category and in each month to show the distribution of categorical variables
  3.3.7 Additional graphics for export - many graphics used in the thesis are generated here
      - Box plots of distribution of goal size in individual categories (both in nominal and real monetary value), also by success
      - Line plot of goal quantiles per category
      - Line plot and scatter plot of the share of succesful project in each goal quantile, also overlayd by bar plot showing the size of the quantiles
      - Line and scatter plot of the share of succesful project in each 1/10 interval of the range of goal, overlaid with the number of observations in the interval
      - Scatter plot of the share of succesful project in each 1/10 interval of the range of goal, size of the point showing number of observations in the interval
      - Line plot of ratio of succesful projects over Campaign Goal quantiles and intervals by category (many versions)
      - Comparison plots showing the estimate of the relationship between the goal and success rate per category using: a) linear trend b) logistic trend c) LOESS
      - Flying box plots of Goal by Category per Year, using real and nominal value of goal and also version including outliers
   3.3.8 Modelling - includes the models used and exports of their outputs to Microsoft Word and quality checks, pointing to the fact that some variables were probably                            missing (visible autocorrelation of residuals) or other misspecification occured
        3.3.8.1 - General model uncompounded (*model (1)*)- Basic model using logistic regression to predict success of a project based on other variables
        3.3.8.2 - General model compounded (*model (2)*)- Based on *model 1*, uses real monetary values
        + ROC to both models *(1)* and *(2)*, VIF calculation
        3.3.8.3.1 - Interactions model uncompounded (*model (3)*)
        
   
