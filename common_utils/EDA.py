from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

def dataset_overview(df):

    print("==================================== Dataset Overview ====================================")
    
    print("")
    
    print("============ Data Shape ============")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("")
    print("")
    print("============ Datatypes ============")
    df.info()

    print("============ Missing Values ============")
    display(df.isnull().sum())

    print("")
    print("")

    print("============ Duplicates Values ============") 
    print(f"Duplicated values : {df.duplicated(keep=False).sum()}")
    if df.duplicated().sum()>0:
        display(df[df.duplicated(keep=False)].reset_index())
    
    print("============ Data Preview ============")
    print("Head:")
    display(df.head(3))
    print("Tail:")
    display(df.tail(3))
    print("Sample:")
    display(df.sample(3))

    print("")
    print("")

    print("============ Numerical and Categorical Values ============")
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"Numerical Datatypes: {num_cols}")
    print(f"Number of numeric features: {len(num_cols)}")
    print(f"Categorical Datatypes: {cat_cols}")
    print(f"Number of categorical features: {len(cat_cols)}")

    return num_cols, cat_cols


def num_analysis(df,col):
    print(f"****************************** {col} analysis ******************************")
    # Plot box and hist plots
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].set_title(f"{col} boxplot")
    axs[0].tick_params(axis='x', rotation=45)
    sns.boxplot(data=df,x=col,ax=axs[0])
    axs[1].set_title(f"{col} histplot")
    axs[1].tick_params(axis='x', rotation=45)

    sns.histplot(data=df,x=col,ax=axs[1],kde=True)
    plt.tight_layout()
    plt.show()

    # Get describe()
    print(f"********************  {col} values description  ********************")
    display(df[col].describe().to_frame().style.background_gradient(cmap='cool'))

    print(f"********************  {col} outliers  ********************")
    # Find upper and lower outliers if any
    Q3 = df[col].quantile(0.75)
    Q1 = df[col].quantile(0.25)

    IQR = Q3 - Q1

    print(f"IQR : {IQR}")

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    upper_outliers = df[df[col] > upper_bound]
    lower_outliers = df[df[col] < lower_bound]

    if len(upper_outliers)>0:
        print(f"****** Upper Outliers ******")
        print(f"Upper outlier count: {len(upper_outliers)}")
        display(upper_outliers.head(3))
    
    if len(lower_outliers)>0:
        print(f"****** Lower Outliers ******")
        print(f"Lower outlier count: {len(lower_outliers)}")
        display(lower_outliers.head(3))

    
    print("")
    print("")
    print("")
    print("")


def categorical_analysis(df,col):
    print(f"****************************** {col} analysis ******************************")
    
    print(f"Number of Unique {col} values: {df[col].nunique()}")
    if df[col].nunique() < 10:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=col, ax=ax, hue=col,legend=False)
        ax.set_title(f"{col} Value Distribution")
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        
        # Add count/percentage labels
        total = df[col].notna().sum()
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height == 0:
                    continue  # Skip empty bars
                count = int(height)
                percentage = f'{100 * height / total:.1f}%'
                ax.annotate(f'{count}\n({percentage})',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Top values for {col}")
        display(df[col].value_counts().reset_index().head(5))

    print("")
    print("")
    print("")
    print("")


