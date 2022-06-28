
import pandas as pd
import time
from scipy.stats import bernoulli, truncnorm
import matplotlib.pyplot as plt


def bernoulli_trial(winrate, num_trials, experiments, index):
    print('bernoulli trial started')
    experiments[index] = bernoulli.rvs(p=winrate, size=num_trials)
    print("completed experiment {} which has {} trials with a winrate of {}".format(
        index, num_trials, winrate))


def generate_experiments(winrate, trials, num_portfolios, experiments):
    for i in range(1, num_portfolios+1):
        bernoulli_trial(winrate=winrate, num_trials=trials,
                        experiments=experiments, index=i)


def generate_df(data, start, gain, gain_std, loss, loss_std):
    df = pd.DataFrame.from_dict(data)
    df.iloc[0] = start
    df[1:] = generate_pnl(df[1:], gain, gain_std, loss, loss_std)
    df = df.cumsum()
    return df


"""
def generate_pnl(df, gain, gain_std, loss, loss_std):
    for col in df.columns.values:
        start_time = time.time()
        print("Generating PNL values for portfolio {}".format(col))
        df.loc[df[col] == 0,col] = df[col].apply(lambda x: -truncnorm.rvs(0, loss+loss_std, loc=loss, scale=loss_std, size=1)[0])
        df.loc[df[col] == 1,col] = df[col].apply(lambda x: truncnorm.rvs(0, gain+gain_std, loc=gain, scale=gain_std, size=1)[0])
        end_time = time.time()
        print("Completed generating PNL values for portfolio {} in {} seconds".format(
            col, end_time - start_time))
    return df
"""


def generate_pnl(df: pd.DataFrame, gain, gain_std, loss, loss_std):
    for col in df.columns.values:
        start_time = time.time()
        print("Generating PNL values for portfolio {}".format(col))
        df[col] = df[col].where(
            (df[col] != 1),
            truncnorm.rvs(
                1, gain+gain_std, loc=gain, scale=gain_std, size=len(df)))
        df[col] = df[col].where(
            (df[col] != 0), -truncnorm.rvs(
                1, loss+loss_std, loc=loss, scale=loss_std, size=len(df)))
        end_time = time.time()
        duration = end_time - start_time
        print(
            f"Completed generating PNL values for portfolio {col} in {duration} seconds")
    return df


def generate_plot(df: pd.DataFrame, start, winrate, gain_amount, loss_amount, num_portfolios, num_trials):
    plot_start = time.time()
    print(
        f"Plotting {num_portfolios} portfolios with starting balances of {start}...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for col in df.columns.values:
        ax.plot(df[col], label=col)
    ax.legend(loc='best')
    ax.set_title("Cumulative Growth of {:d} ${:.2f} portfolios with a {:.2f}% winrate ({:.2f} pay ratio) over {:d} trials".format(
        num_portfolios, start, winrate*100, gain_amount/loss_amount, num_trials))
    ax.set_xlabel('Trials')
    ax.set_ylabel('Portfolio Value')
    fig.savefig('Cumulative_Growth_of_Portfolios.png')
    plot_end = time.time()
    plot_duration = plot_end - plot_start
    print(f"Created plots which took {plot_duration} seconds to generate")


def generate_stats(df, rolling_days):
    stats = {}
    for col in df.columns.values:
        max_return = df[col].pct_change().max()
        min_return = df[col].pct_change().min()
        weekly_mean = df[col].pct_change().rolling(
            rolling_days).mean().mean()
        weekly_std = df[col].pct_change().rolling(rolling_days).mean().std()
        total_mean = df[col].pct_change().mean()
        total_std = df[col].pct_change().std()
        stats[col] = [max_return, min_return, weekly_mean,
                      weekly_std, total_mean, total_std]
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=["Maximum Return", "Minimum Return",
                                      '{} day mean return'.format(rolling_days), '{} day volatility'.format(rolling_days), 'Total Portfolio Mean Return', 'Total Portfolio Volatility'])
    stats_df.index.rename("Portfolio", inplace=True)
    stats_df = stats_df.round(5)
    return stats_df


def validate_start():
    while True:
        try:
            balance = float(
                input("Please enter your starting portfolio amount (greater than 0): "))
            if(balance <= 0):
                print("Please enter a valid portfolio balance")
                continue
            else:
                return balance
        except ValueError:
            print('You must enter a valid price.')


def validate_winrate():
    while True:
        try:
            win_rate = float(
                input("Please enter your desired win rate (0-1): "))
            if(win_rate <= 0 or win_rate > 1):
                print("Please enter a win ratio")
                continue
            else:
                return win_rate
        except ValueError:
            print('You must enter a valid win ratio.')


def validate_gain():
    while True:
        try:
            gain = float(
                input("Please enter your average gain amount (greater than 0): "))
            if(gain < 0):
                print("Please enter a gain amount that is greater than 0")
                continue
            else:
                return gain
        except ValueError:
            print('You must enter a valid gain amount.')


def validate_loss():
    while True:
        try:
            loss = float(
                input("Please enter your average loss amount (greater than 0): "))
            if(loss < 0):
                print("Please enter a loss amount that is greater than 0")
                continue
            else:
                return loss
        except ValueError:
            print('You must enter a valid loss amount.')


def validate_gain_std():
    while True:
        try:
            gain = float(
                input("Please enter your average gain std amount (greater than 0): "))
            if(gain < 0):
                print(
                    "Please enter a gain std amount that is greater than or equal to 0")
                continue
            else:
                return gain
        except ValueError:
            print('You must enter a valid gain std amount.')


def validate_loss_std():
    while True:
        try:
            loss = float(
                input("Please enter your average std loss amount (greater than 0): "))
            if(loss < 0):
                print("Please enter a loss std amount that is greater than 0")
                continue
            else:
                return loss
        except ValueError:
            print('You must enter a valid loss std amount.')


def validate_trials():
    while True:
        try:
            num_trials = int(input(
                "Please enter the amount of trials per experiment (greater than or equal to 50): "))
            if(num_trials < 50):
                print("Please enter a trial amount that is greater than 50")
                continue
            else:
                return num_trials
        except ValueError:
            print('You must enter a valid trial amount.')


def validate_portfolios():
    while True:
        try:
            num_portfolios = int(
                input("Please enter the amount of experiments (greater than 0): "))
            if(num_portfolios <= 0):
                print("Please enter a number of portfolios greater than 0")
                continue
            else:
                return num_portfolios
        except ValueError:
            print('You must enter a valid number of portfolios.')


def validate_rolling_days(trials):
    while True:
        try:
            rolling_days = int(
                input("Please enter the rolling window (less than the number of trials): "))
            if(rolling_days >= trials):
                print(
                    "Please enter a valid rolling window that is less than number of trials")
                continue
            else:
                return rolling_days
        except ValueError:
            print('You must enter a valid rolling window')


def validate_entries():
    begin_balance = validate_start()
    win_rate = validate_winrate()
    gain_amount = validate_gain()
    gain_std = validate_gain_std()
    loss_amount = validate_loss()
    loss_std = validate_loss_std()
    num_trials = validate_trials()
    num_portfolios = validate_portfolios()
    rolling_days = validate_rolling_days(num_trials)
    return begin_balance, win_rate, gain_amount, gain_std, loss_amount, loss_std, num_trials, num_portfolios, rolling_days


def main():
    while True:
        flag1 = str(input("Press any key to start (Press q to quit): "))
        if flag1 == 'q':
            break
        begin_balance, win_rate, gain_amount, gain_std, loss_amount, loss_std, num_trials, num_portfolios, rolling_days = validate_entries()
        experiments = {}
        begin = time.time()
        generate_experiments(winrate=win_rate, trials=num_trials,
                             num_portfolios=num_portfolios, experiments=experiments)
        end = time.time()
        print("Generated {} experiments which took approx {} seconds to complete".format(
            num_portfolios, end - begin))

        df = generate_df(experiments, begin_balance, gain_amount,
                         gain_std, loss_amount, loss_std)
        print(generate_stats(df, rolling_days))

        generate_plot(df, begin_balance, win_rate, gain_amount,
                      loss_amount, num_portfolios, num_trials)
        num_data = num_portfolios * num_trials
        print(f"Creating \"data.csv\" with over {num_data} points of data")
        csv_start = time.time()
        df.to_csv('data.csv', index=True)
        csv_end = time.time()
        csv_duration = csv_end - csv_start
        print(f"Created csv file which took {csv_duration} to generate")


if __name__ == "__main__":
    main()
