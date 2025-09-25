SELECT 
          leh.lot AS lot
         ,wch.waf3 AS waf3
         ,wch.wafer AS waferid
         ,leh.entity AS entity
         ,wch.slot AS slot
         ,wch.chamber AS chamber
         ,To_Char(leh.introduce_txn_time,'yyyy-mm-dd hh24:mi:ss') AS intro_date
         ,wch.state AS state
         ,To_Char(wch.start_time,'yyyy-mm-dd hh24:mi:ss') AS start_date
         ,To_Char(wch.end_time,'yyyy-mm-dd hh24:mi:ss') AS end_date
         ,lwr2.recipe AS recipe
         ,leh.lot_abort_flag AS abort_flag
         ,leh.load_port AS load_port
         ,leh.processed_wafer_count AS qty
         ,leh.reticle AS reticle
         ,lrc.rework AS rework
         ,leh.operation AS opn
         ,leh.route AS route
         ,leh.product AS product
         ,wch.chamber_wait_duration AS chamber_wait_duration
         ,wch.chamber_sequence AS chamber_sequence
         ,wch.chamber_process_order AS chamber_process_order
         ,wch.chamber_process_duration AS chamber_process_duration
FROM 
F_LotEntityHist leh
INNER JOIN
F_WaferChamberHist wch
ON leh.runkey = wch.runkey
INNER JOIN F_Lot_Wafer_Recipe lwr2 ON lwr2.recipe_id=leh.lot_recipe_id
INNER JOIN F_Lot_Run_card lrc ON lrc.lotoperkey = wch.lotoperkey
WHERE
              (wch.chamber LIKE  '%')
 AND      (leh.entity LIKE 'SCJ591%')
 AND      (leh.lot LIKE  'W%') 
 AND      lwr2.recipe Like '%' 
 AND      wch.start_time >= TRUNC(SYSDATE) - 14