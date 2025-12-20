## Multilingual Extension (Optional)

The proposed AutoCodeDataPipeline is language-agnostic by design.

All language-dependent components are isolated in the template layer, including:
- Question/answer templates for QA generation
- Requirement templates for design task generation
- Natural language descriptions in reasoning traces

Core stages such as:
- repository indexing and chunking,
- domain mapping,
- rule and flow extraction,
- evidence grounding,
- trace construction,
remain entirely independent of natural language.

Therefore, by switching or extending the template files (e.g., to English or bilingual templates),
the same pipeline can be reused to generate multilingual training datasets without modifying
any core extraction or validation logic.

This design allows the pipeline to be easily adapted for multilingual instruction tuning
or cross-lingual code understanding tasks.
