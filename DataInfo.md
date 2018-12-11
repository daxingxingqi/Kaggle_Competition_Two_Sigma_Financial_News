### 竞赛数据包含两大部分：
###### 市场金融信息-MaketData
###### 新闻文章-News
#### MaketData，News中采用asset Conde作为资产的标志，并且一个公司可以有多个资产。
#### 数据格式采用pandas，数据存在间隙（不一定意味着数据不存在，由于选择标准，这些行可能不包括在内）
### MaketData包括：
###### time 当前时间
###### assetCode 资产ID
###### assetName 对应assetCode的名称
###### universe 
###### volume 股票交易量
###### close 收盘价（未调整 分割或股息）
###### open 未平仓价格（未调整 分割或股息）
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
######  *1：标题*
######  *2：第一句话*
######  *3：第二句等*
######  *0：在新闻项目的标题或正文中找不到正在评分的资产。结果，整个新闻项目的文本（标题+正文）将用于确定情绪分数。*
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
