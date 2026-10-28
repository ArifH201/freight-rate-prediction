const fs=require('fs'); const {Document,Packer,Paragraph,TextRun,HeadingLevel,ImageRun,Table,TableRow,TableCell,WidthType,ShadingType,AlignmentType,BorderStyle,LevelFormat}=require('docx');
const img=(p,w,h)=>new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:80,after:80},children:[new ImageRun({type:'png',data:fs.readFileSync(p),transformation:{width:w,height:h}})]});
const P=(t,o={})=>new Paragraph({spacing:{after:100},children:Array.isArray(t)?t:[new TextRun(t)],...o});
const B=(t)=>new Paragraph({numbering:{reference:'b',level:0},spacing:{after:60},children:Array.isArray(t)?t:[new TextRun(t)]});
const b=(t)=>new TextRun({text:t,bold:true}); const r=(t)=>new TextRun(t);
const H1=(t)=>new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:220,after:100},children:[new TextRun(t)]});
const cell=(t,w,o={})=>new TableCell({width:{size:w,type:WidthType.DXA},margins:{top:50,bottom:50,left:90,right:90},
  shading:o.head?{fill:'064A56',type:ShadingType.CLEAR,color:'auto'}:(o.hl?{fill:'E3F1F3',type:ShadingType.CLEAR,color:'auto'}:undefined),
  children:[new Paragraph({children:[new TextRun({text:t,bold:!!(o.head||o.hl),color:o.head?'FFFFFF':undefined,size:19})]})]});
const bd={style:BorderStyle.SINGLE,size:4,color:'C9D6D9'}; const borders={top:bd,bottom:bd,left:bd,right:bd,insideHorizontal:bd,insideVertical:bd};
const table=(cols,rows,hl=-1)=>new Table({width:{size:cols.reduce((a,c)=>a+c,0),type:WidthType.DXA},columnWidths:cols,borders,
  rows:rows.map((row,i)=>new TableRow({children:row.map((t,j)=>cell(t,cols[j],{head:i===0,hl:i===hl}))}))});

const children=[
 new Paragraph({spacing:{after:40},children:[new TextRun({text:'Freight Rate Prediction – Approach & Validation Report',bold:true,size:36,color:'064A56'})]}),
 P([new TextRun({text:'Machine Learning Engineer assessment  |  Code: GitHub repository (see README)',color:'455A60',size:20})]),

 H1('1. Summary'),
 P('The task is to predict posted_rate for 12,000 loads dated Nov–Dec 2025 using 48,000 labelled loads dated Jan–Oct 2025. Because the final predictions are for the future, I validated with forward-chaining (train on earlier months, test on the next month) instead of a random split. The chosen model is a LightGBM lane model (L1 loss on log-rate) with a small explicit market adjustment. It deliberately does not use quote_signal, which I found to be unreliable across time.'),
 table([5400,1320,1320,1320],[['Forward-CV, mean of 5 folds (Jun–Oct)','MAE ($)','MAPE','MAPE excl. spikes'],
   ['Naive: distance × median $/mile by equipment','225.3','10.4%','7.8%'],
   ['LightGBM lane model, no date information','129.1','5.3%','3.1%'],
   ['+ recent price level (last 30 days)','119.7','5.1%','2.7%'],
   ['FINAL: + market_index adjustment','113.5','4.8%','2.5%']],4),
 P([new TextRun({text:'“Excl. spikes” drops test rows whose label is >30% away from the prediction (≈1.4% of rows, see §2), which no feature in the data can explain.',italics:true,size:18,color:'455A60'})]),

 H1('2. Data exploration and data-quality issues'),
 B([b('Temporal shift by design. '),r('train_test.csv covers 2025-01-01 to 2025-10-31; validation.csv covers 2025-11-01 to 2025-12-31. Any honest validation has to extrapolate forward in time.')]),
 B([b('quote_signal is not trustworthy (key finding). '),r('In Jan–Mar, Jun and Sep it is almost exactly rate ÷ distance (within 2% for ~93% of rows – effectively a leak). In Apr, May, Jul and Oct that relationship is inverted (positively correlated with distance), and in Aug it is pure noise. In the validation months it is uncorrelated with distance (r≈0.01), i.e. it looks like the August regime. A model that learns to trust it breaks when it flips: on the August test fold, adding quote_signal raised MAE from $99.7 to $149.4. I excluded it.')]),
 img('outputs/fig_quote_signal.png',560,241),
 B([b('Weight: '),r('292 negative values in train (145 in validation) – sign errors, fixed with abs(); 300 missing (165 in validation) – left as NaN for LightGBM; 1,204 train rows sit exactly at the 47,500 lb cap (censored), flagged with an indicator.')]),
 B([b('market_index: '),r('374 missing in train (249 in validation). It is a daily series, so missing values are filled with the same-date median across loads.')]),
 B([b('Unseen cities: '),r('validation has 72 cities, train 64. Eight cities (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego) never appear in training (~12% of validation rows). I therefore use latitude/longitude, not city names or lane IDs. Predicted $/mile for unseen-city lanes (mean 2.19) matches seen-city lanes (2.19).')]),
 B([b('Label spikes: '),r('about 1.4% of training rates are far from the model (about half far too high, half far too low; rate-per-mile up to ~14). These look like entry errors/extreme outliers. I use an L1 (median) loss so they have little pull; explicitly removing them from training changed CV MAE by <0.1, so the robust loss is sufficient. Residual MAE/RMSE is dominated by them (RMSE ≈ $644 regardless of model).')]),
 B([b('Other: '),r('no duplicate IDs or rows; distance has a 70-mile floor (48 rows) and runs ~1.18× straight-line distance (median) – kept as is; rate-per-mile falls with distance, and weight, equipment (Reefer > Flatbed > Dry Van) and region (Texas/Oklahoma higher, Northeast lower) all matter.')]),
 B([b('Date effect: '),r('after removing lane effects, the daily median log-rate residual is smooth (lag-1 autocorrelation 0.96) and tracks market_index (r=0.71 daily; R²=0.86 with a linear trend). Nov–Dec market_index (≈0.93) is close to October’s (≈0.96), so no large level change is expected.')]),

 H1('3. Train / validation split approach'),
 P('Random K-fold would mix future and past rows, let the model learn month-specific quirks (such as the quote_signal regime) and overstate accuracy. I used forward-chaining: for k = June…October, train on all months before k and score on month k. Every fold therefore predicts a month whose date-level conditions were never seen, which is exactly the situation for Nov–Dec. Fold results for the final model:'),
 img('outputs/fig_split.png',560,210),
 table([1500,1500,1500,1500,1800],[['Test month','MAE ($)','MAPE','MedAPE','MAPE excl. spikes'],
   ['June','150.1','5.4%','3.2%','3.5%'],['July','108.3','4.6%','2.0%','2.4%'],['August','89.0','4.2%','1.5%','1.8%'],['September','120.4','4.9%','2.5%','2.9%'],['October','99.7','4.8%','1.4%','1.8%']]),
 P([new TextRun({text:'June is hardest because it follows a sharp rise in the price level. LightGBM hyper-parameters were left at sensible defaults and not tuned. The date-adjustment choices (30-day window, using market_index) were compared on these same folds, so the CV numbers are slightly optimistic. The final model is refit on all 48,000 labelled rows.',size:19})],{spacing:{before:80,after:100}}),

 H1('4. Model choice and reasoning'),
 B([b('Lane model: '),r('gradient-boosted trees on log(rate), L1 loss. Rates are multiplicative in distance/equipment/weight, trees capture the non-linear $/mile-vs-distance curve and regional interactions automatically, and L1 on the log scale optimises median percentage error while ignoring label spikes.')]),
 B([b('No raw date features: '),r('trees cannot extrapolate to unseen months (they reuse the last leaf), and a raw date lets the model memorise period-specific noise.')]),
 B([b('Explicit date adjustment: '),r('prediction = lane model × exp(level + β·(market_index − recent market_index)). “level” is the median residual of the last 30 training days (it keeps the model at the current price level) and β = 0.131 is fitted on the daily residual series. This simple term cut CV MAE from $129 to $114 and is easy to explain and sanity-check.')]),
 B([b('Rejected: '),r('quote_signal (regime-dependent, see §2); raw market_index inside the GBM (it extrapolated badly outside its trained range); linear trend extrapolation (unstable across folds).')]),

 H1('5. Fixed December prediction chart'),
 P('Produced by the provided score.py from the completed december_chart_inputs.csv (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb, only the date varies). The model needs a market_index for each day; I used the same-day median market_index of the validation loads (an input feature, not a label). Day-to-day variation comes only from that market_index series; predictions range $819–$843 (≈$2.28–$2.34/mile).'),
 img('outputs/candidate_december.png',560,182),

 H1('6. Limitations'),
 B('Nov–Dec rate level is extrapolated from the last 30 days plus market_index; a regime change not visible in market_index would not be captured (June’s fold error shows the size of that risk).'),
 B('Spike labels (~1.4%) are unpredictable from the supplied features; if the final metric is RMSE-based it will be dominated by them for any model.'),
 B('The model predicts the conditional median. If the scoring metric rewarded the conditional mean, a slight upward bias correction (~1–2%) could help; this was not applied since the metric is not specified.'),
];
const doc=new Document({styles:{default:{document:{run:{font:'Calibri',size:21}}},paragraphStyles:[{id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:27,bold:true,color:'064A56',font:'Calibri'},paragraph:{spacing:{before:220,after:100},outlineLevel:0}}]},
 numbering:{config:[{reference:'b',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:500,hanging:260}}}}]}]},
 sections:[{properties:{page:{size:{width:12240,height:15840},margin:{top:1000,bottom:1000,left:1200,right:1200}}},children}]});
Packer.toBuffer(doc).then(buf=>fs.writeFileSync('outputs/Freight_Rate_Report.docx',buf));
