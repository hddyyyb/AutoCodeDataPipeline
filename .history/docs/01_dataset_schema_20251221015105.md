domain_map.json结构：

repo:{repo_path,generated_at,source_index}

boundaries:{controller,service,mapper,model,config,other}每个是chunk_id列表

entities:[{name,domain,confidence,evidence_chunks,mentions}]

operations:[{name,domain,confidence,evidence_chunks,signals}]

candidate_flows:[{flow_id,name,domain,steps:[{idx,operation,evidence_chunk,why}],evidence_chunks}]