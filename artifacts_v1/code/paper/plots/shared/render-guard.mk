LIN_SAVED_DEFAULT_GOAL := $(.DEFAULT_GOAL)
.PHONY: render-guard
render-guard:
ifneq ($(LIN_RENDER),1)
	@echo "figures are rendered from fleet data; set LIN_RENDER=1 if you really mean it" >&2; exit 1
else
	@:
endif
.DEFAULT_GOAL := $(LIN_SAVED_DEFAULT_GOAL)
