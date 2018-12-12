### 竞赛数据包含两大部分：
###### 市场金融信息-MaketData
###### 新闻文章-News
#### `MaketData`，`News`中采用 `asset Conde` 作为资产的标志，并且一个公司可以有多个资产。
#### 数据集使用合成市场和新闻数据填充的，此合成数据旨在模拟真实的未来数据将引入的数量，时间线和计算负担。需要注意此合成数据不遵守交易日历，并且{`time`,`assetCode`}组合的组成在实际数据中会有所不同。
##### 代码基本格式`train_my_model`和`make_my_predictions`需要实现的部分.[原始链接](https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel)
    from kaggle.competitions import twosigmanews
    env = twosigmanews.make_env()
    (market_train_df, news_train_df) = env.get_training_data()
    train_my_model(market_train_df, news_train_df)
    for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
        predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
        env.predict(predictions_df)
    env.write_submission_file()
##### 只能迭代一次 `get_prediction_days()`
##### 你开始迭代时小心不要忘记调用一次
    days = env.get_prediction_days()
##### `market_observations_df`: 下一个预测日的市场观察
##### `news_observations_df`: 下一个预报日的新闻观察
##### `predictions_template_df`: 带有`assetCode`和`confidenceValue`列的`DataFrame`，前缀有`confidenceValue=0`，用于填充并传递回预测函数
    (market_obs_df, news_obs_df, predictions_template_df) = next(days)
##### 循环所有的日子，做出我们的随机预测
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        make_random_predictions(predictions_template_df)
        env.predict(predictions_template_df)
    print('Done!')
##### 进行随机预测一天
    def make_random_predictions(predictions_df):
        predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
##### 将预测写入当前工作目录中的CSV文件（submission.csv）
    env.write_submission_file()
    print([filename for filename in os.listdir('.') if '.csv' in filename])
#### 数据格式采用`pandas`，数据存在间隙（不一定意味着数据不存在，由于选择标准，这些行可能不包括在内）
### MaketData包括：
###### time 当前时间（所有行均在UTC时间22：00获取）
###### assetCode 资产ID
###### assetName 对应assetCode的名称
###### universe 表示当天的消息评分是否包含在评分中
##### [股票评分例子](http://www.sohu.com/a/113545796_379325)
###### volume 股票交易量
##### 开盘是指某种证券在证券交易所每个营业日的第一笔交易，第一笔交易的成交价即为当日开盘价。按上海证券交易所规定，如开市后半小时内某证券无成交，则以前一天的盘价为当日开盘价。有时某证券连续几天无成交，则由证券交易所根据客户对该证券买卖委托的价格走势，提出指导价格，促使其成交后作为开盘价。
##### 收盘价是指某种证券在证券交易所一天交易活动结束前最后一笔交易的成交价格。如当日没有成交，则采用最近一次的成交价格作为收盘价，因为收盘价是当日行情的标准，又是下一个交易日开盘价的依据，可据以预测未来证券市场行情；所以投资者对行情分析时，一般采用收盘价作为计算依据。
###### close 收盘价（未调整分割或股息）
###### open 未平仓价格（未根据拆分或股息进行调整）
##### 平仓是指交易者了结持仓的交易行为，了结的方式是针对持仓方向作相反的对冲买卖
##### 股票回报率是将股票投资的盈利除以投入资金的平均数字而计算出来的。预测股票回报率比预测天气更加困难。天气预报虽也受诸多因素影响而难以准确预测，但有一点，是天气会按自己的规律运行，不会因为人们的预报而发生变化。相反，股票回报率会因为对它的预测而发生改变。比如，人们预测股票甲的回报率会超过股票乙的回报率，假如这个预测是准确的话，那么，人们在得知预测后，会立刻采取行动，购买股票甲，抛掉股票乙。同对对于已经持有股票甲的人，更不会轻易抛掉这只股。买者多，卖者少，股票甲的成交量低而价格会迅速上升，远超过它的价值。这时如果在很高的价位买入股票甲的话，未来在价格重新调整回价值时，它的回报率极有可能是负值。相反，对于股票乙，由于人们都急于抛售，价格会迅速降低，以至于低于它的价值，未来在价格重新调整回价值时，股票乙的回报率可能是正值。股票乙的回报率反而高于股票甲的回报率，走向与预测完全相反。从而使一个准确的预测，变成了一个无效的预测。人们根据预测产生行动，从而改变了预测的结果。这种现象在经济学上称为“自指性”(Self- referentiality)。
###### returnsClosePrevRaw1 1天后的收盘市场原始回报
###### returnsOpenPrevRaw1 1天后的未平仓市场原始回报
###### returnsClosePrevRaw10 10天后的收盘市场原始回报
###### returnsOpenPrevRaw10 10天后的未平仓市场原始回报
###### returnsClosePrevMktres1 1天后的收盘市场残余回报
###### returnsOpenPrevMktres1 1天后的未平仓市场残余回报
###### returnsClosePrevMktres10 10天后的收盘市场残余回报
###### returnsOpenPrevMktres10 10天后的未平仓的市场残余回报
###### returnsOpenNextMktres10 10天前的未平仓的市场残余回报
### News包括：
###### time 显示在Feed上的时间戳
###### sourceTimestamp 新闻项创建的时间戳
###### firstCreated 项目第一个版本的时间戳
###### sourceId 新闻项的ID
###### headline 项目的标题
###### urgency 区分类型（1警报，3文章）
###### takeSequence 新闻项的获取序列号，警报和文章有单独的序列
###### provider 新闻提供组织的标识符
###### subjects 与此新闻项相关的主题代码和公司标识符。主题代码描述了新闻项目的主题。这些可以涵盖资产类别，地理位置，事件，行业/部门和其他类型
###### audiences 针对特定受众量身定制（例如，国际新闻服务的“M”和法国综合新闻服务的“FB”）
###### bodySize 以字符为单位的故事主体的当前版本的大小
###### companyCount 在主题字段中的新闻项中明确列出的公司数量
###### headlineTag
###### marketCommentary
###### sentenceCount 新闻项中的句子总数
###### wordCount 新闻中的标点符号的总数
###### assetCodes 项目中提到的资产清单
###### assetName 资产的名称
###### firstMentionSentence 从标题开始的第一个句子，其中提到了评分资产
######  > *1：标题*
######  > *2：第一句话*
######  > *3：第二句等*
######  > *0：在新闻项目的标题或正文中找不到正在评分的资产。结果，整个新闻项目的文本（标题+正文）将用于确定情绪分数。*
###### relevance 表示新闻项与资产的相关性。它的范围从0到1.如果在标题中提到资产，则相关性设置为1.当项目是警报（紧急度== 1）时，相关性应由firstMentionSentence来衡量
###### sentimentClass 表示此新闻项相对于资产的主要情绪类。指示的类是具有最高概率的类。
###### sentimentNegative 新闻项目的情绪对资产为负的概率
###### sentimentNeutral 新闻项目的情绪对资产是中性的概率
###### sentimentPositive 新闻项目的情绪对资产有利的概率
###### sentimentWordCount 项目文本中与资产相关的部分中的词汇标记数。这可以与wordCount结合使用，以确定讨论资产的新闻项目的比例
###### noveltyCount12H 特定资产的新闻项目内容的12小时新颖性。它是通过将其与包含资产的先前新闻项的缓存中的资产特定文本进行比较来计算的。
###### noveltyCount24H 特定资产的新闻项目内容的24小时新颖性。
###### noveltyCount3D 特定资产的新闻项目内容的3天新颖性。
###### noveltyCount5D 特定资产的新闻项目内容的5天新颖性。
###### noveltyCount7D 特定资产的新闻项目内容的7天新颖性。
###### volumeCounts12H 每项资产的12小时新闻量。保持先前新闻项的缓存，并计算在五个历史时段的每一个中提及资产的新闻项的数量
###### volumeCounts24H 每项资产的24小时新闻量。
###### volumeCounts3D 每项资产的3天新闻量。
###### volumeCounts5D 每项资产的5天新闻量。
###### volumeCounts7D 每项资产的7天新闻量。
